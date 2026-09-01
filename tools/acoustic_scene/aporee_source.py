"""License-audited Radio Aporee field recordings for scene training.

Radio Aporee publishes each field recording as a separate Internet Archive
item.  The first numeric component of an item id identifies the map location,
so it is also the leakage boundary used by the product train/calibration/test
protocol.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import quote

import requests


ARCHIVE_SEARCH_URL = "https://archive.org/advancedsearch.php"
ARCHIVE_METADATA_URL = "https://archive.org/metadata/{identifier}"
ARCHIVE_DOWNLOAD_URL = "https://archive.org/download/{identifier}/{filename}"

# A product scene is intentionally a 30-second decision.  Every selected
# field recording contributes three adjacent ten-second windows so temporal
# smoothing is evaluated on real continuity rather than unrelated clips.
CONTINUOUS_SCENE_OFFSETS_SECONDS = (0, 10, 20)

APOREE_QUERIES = {
    "high_speed_train": (
        '(title:(TGV OR Shinkansen OR KTX OR THSR OR Eurostar OR "ICE train" '
        'OR "high speed train" OR "high-speed train" OR "high speed railway" '
        'OR "high-speed rail") OR description:(TGV OR Shinkansen OR KTX OR '
        'THSR OR Eurostar OR "ICE train" OR "high speed train" OR '
        '"high-speed train" OR "high speed railway" OR "high-speed rail"))'
    ),
    "metro": (
        '(title:(metro OR subway OR "underground train" OR "tube station") '
        'OR description:(metro OR subway OR "underground train" OR '
        '"tube station"))'
    ),
    "shopping_mall": (
        '(title:("shopping mall" OR "shopping centre" OR "shopping center" '
        'OR "department store" OR IKEA OR "furniture store") OR '
        'description:("shopping mall" OR "shopping centre" OR '
        '"shopping center" OR "department store" OR IKEA OR '
        '"furniture store"))'
    ),
    "cafe_restaurant": (
        '(title:(cafe OR café OR restaurant OR "coffee shop") OR '
        'description:(cafe OR café OR restaurant OR "coffee shop"))'
    ),
    "concert": (
        '(title:(concert OR "live music" OR "music festival" OR recital OR gig) '
        'OR description:(concert OR "live music" OR "music festival" OR '
        'recital OR gig))'
    ),
}

# Description-only matches frequently mention a nearby place rather than the
# recorded scene.  Requiring the evidence in the title makes the automatic
# subset conservative and leaves ambiguous recordings out of all splits.
TITLE_EVIDENCE = {
    "high_speed_train": re.compile(
        r"\btgv\b|shinkansen|\bktx\b|\bthsr\b|eurostar|"
        r"\bice ?[1-4]?\b.{0,30}(train|station|route)|"
        r"high[- ]speed (train|rail|railway)",
        re.IGNORECASE,
    ),
    "metro": re.compile(
        r"\bmetro\b|\bsubway\b|underground (train|station)|\btube station\b",
        re.IGNORECASE,
    ),
    "shopping_mall": re.compile(
        r"shopping (mall|centre|center)|department store|\bikea\b|"
        r"furniture store",
        re.IGNORECASE,
    ),
    "cafe_restaurant": re.compile(
        r"\bcafe\b|\bcafé\b|\brestaurant\b|coffee shop",
        re.IGNORECASE,
    ),
    "concert": re.compile(
        r"\bconcert\b|live music|music festival|\brecital\b|\bgig\b",
        re.IGNORECASE,
    ),
}

TITLE_REJECTION = {
    "high_speed_train": re.compile(
        r"taxi ride to the shinkansen|parking|hotel|birds? with high[- ]speed|"
        r"cicada.{0,24}high[- ]speed|outside (the )?station|station square|"
        r"square (of|outside) the station|under the high[- ]speed|"
        r"near (the )?(high[- ]speed|railway|tunnel)|next to the high[- ]speed|"
        r"high voltage line|bus station|internal airport shuttle|"
        r"elevator, station ambience",
        re.IGNORECASE,
    ),
    "metro": re.compile(
        r"metropolitan (museum|area)|metro[- ]goldwyn|metronom|"
        r"near (the )?.{0,24}(metro|subway)|not far from.{0,24}metro|"
        r"between.{0,32}subway station|outside.{0,24}metro|"
        r"in front of.{0,24}(metro|subway)|square near subway|"
        r"park near.{0,24}metro|metrobus|construction work.{0,24}subway|"
        r"metro entrance|subway station entrance|entrance to metro|"
        r"entrance centrum metro|toward the subway station|"
        r"tunnel for bikers|cars, metro trains|capital metro - freeride",
        re.IGNORECASE,
    ),
    "shopping_mall": re.compile(
        r"parking lot|parking garage|car park|outside|in front of|"
        r"basement|underground tour|automatic glass door|fireworks|"
        r"near by the shopping|at ikea entrance|steak house|"
        r"old department store",
        re.IGNORECASE,
    ),
    "cafe_restaurant": re.compile(
        r"street caf|roadside restaurant|restaurant car|outside|"
        r"in front of|next to the restaurant|near the caf|nearby.{0,24}restaurant|"
        r"at the door of|entrance room|terrace|patio|outdoor caf|"
        r"restaurant garden|cafe gardens|restaurant kitchen|cafe kitchen|"
        r"in the back of.{0,24}restaurant|restaurant kitchen window|"
        r"hummingbirds|fireworks|pebble beach|restaurant la d.dicace|"
        r"accordionist plays|fountain and a cafe",
        re.IGNORECASE,
    ),
    "concert": re.compile(
        r"pre[- ]concert|post[- ]concert|before (the )?concert|"
        r"minutes before|playground in concert hall|far away concert|"
        r"after a concert|concert intermission|concert hall.{0,24}waiting|"
        r"concert hall.{0,24}plaza|excerpt of bird concert|cicada concert|"
        r"nature night concert|recital hall|ships? horns? concert|frog concert|"
        r"distant concert|rhythmic concert|carillon.{0,24}avant le concert|"
        r"four minutes and thirty|bell concert|snoring concert|"
        r"conversations.{0,24}before a concert|avant concert|concert de chiens|"
        r"city birds concert|rooster.s morning concert|salle de concert|"
        r"concert d.oiseaux|concert de grillons|concert de merles|"
        r"concert matinal|concert nearby|bells and dog",
        re.IGNORECASE,
    ),
}

# Only unequivocal rejected titles become ``other`` hard negatives. Ambiguous
# boundary cases (for example a station entrance or an outdoor cafe) stay out
# of every split instead of receiving a convenient but debatable label.
HARD_NEGATIVE_TITLE = {
    "high_speed_train": re.compile(
        r"birds? with high[- ]speed|cicada.{0,24}high[- ]speed|"
        r"outside (the )?station|station square|square (of|outside) the station|"
        r"under the high[- ]speed|near (the )?(high[- ]speed|railway)|"
        r"next to the high[- ]speed|high voltage line|bus station|"
        r"internal airport shuttle",
        re.IGNORECASE,
    ),
    "metro": re.compile(
        r"near (the )?.{0,24}(metro|subway)|not far from.{0,24}metro|"
        r"between.{0,32}subway station|outside.{0,24}metro|"
        r"in front of.{0,24}(metro|subway)|square near subway|"
        r"park near.{0,24}metro|metrobus|construction work.{0,24}subway|"
        r"tunnel for bikers|cars, metro trains|capital metro - freeride",
        re.IGNORECASE,
    ),
    "shopping_mall": re.compile(
        r"parking lot|parking garage|car park|old department store|"
        r"underground tour|fireworks|near by the shopping",
        re.IGNORECASE,
    ),
    "cafe_restaurant": re.compile(
        r"pebble beach|hummingbirds|fireworks|in the back of.{0,24}restaurant|"
        r"near the caf.{0,24}tamanno",
        re.IGNORECASE,
    ),
    "concert": re.compile(
        r"bird concert|cicada concert|nature night concert|ships? horns? concert|"
        r"frog concert|rhythmic concert|snoring concert|concert de chiens|"
        r"city birds concert|rooster.s morning concert|concert d.oiseaux|"
        r"concert de grillons|concert de merles|concert matinal|concert nearby",
        re.IGNORECASE,
    ),
}

WEAK_POSITIVE_METADATA = {
    "high_speed_train": re.compile(
        r"in the square outside the station|outside the station|station square",
        re.IGNORECASE,
    ),
    "metro": re.compile(r"$^"),
    "shopping_mall": re.compile(
        r"shopping (center|centre).{0,32}laundry|department store.{0,64}basement",
        re.IGNORECASE,
    ),
    "cafe_restaurant": re.compile(
        r"desert wind|p.ssaros no caf|birds?.{0,24}(at|in).{0,12}caf|"
        r"before entering.{0,24}restaurant|airport.{0,80}caf|"
        r"caf.{0,32}(in|inside).{0,24}supermarket|bakery caf.{0,24}supermarket",
        re.IGNORECASE,
    ),
    "concert": re.compile(
        r"avant le concert|before the concert|concert in 5 minutes|"
        r"concert intermission|waiting to enter the concert",
        re.IGNORECASE,
    ),
}

COMMERCIAL_LICENSE_MARKERS = (
    "creativecommons.org/publicdomain/mark/",
    "creativecommons.org/publicdomain/zero/",
    "creativecommons.org/licenses/by/",
)


@dataclass(frozen=True)
class AporeeCandidate:
    identifier: str
    label: str
    title: str
    description: str
    creator: str
    license: str
    location_group: str


def _request_json(url: str, *, params: dict[str, object] | None = None) -> object:
    for attempt in range(8):
        try:
            response = requests.get(url, params=params, timeout=(30, 120))
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", "0") or 0)
                time.sleep(max(retry_after, min(60, 2 ** attempt)))
                continue
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError):
            if attempt == 7:
                raise
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"unable to read JSON from {url}")


def _location_group(identifier: str) -> str:
    match = re.fullmatch(r"aporee_(\d+)_\d+", identifier)
    if match is None:
        raise ValueError(f"unexpected Radio Aporee identifier: {identifier}")
    return f"aporee-location:{match.group(1)}"


def is_commercial_license(license_url: str) -> bool:
    return any(marker in license_url for marker in COMMERCIAL_LICENSE_MARKERS)


def candidate_from_document(
    label: str, document: dict[str, object]
) -> AporeeCandidate | None:
    identifier = str(document.get("identifier", ""))
    title = str(document.get("title", ""))
    license_url = str(document.get("licenseurl", ""))
    if not identifier.startswith("aporee_") or not is_commercial_license(license_url):
        return None
    if TITLE_EVIDENCE[label].search(title) is None:
        return None
    creator_value = document.get("creator", "Radio Aporee contributor")
    if isinstance(creator_value, list):
        creator = ", ".join(str(value) for value in creator_value)
    else:
        creator = str(creator_value)
    description_value = document.get("description", "")
    if isinstance(description_value, list):
        description = " ".join(str(value) for value in description_value)
    else:
        description = str(description_value)
    combined_metadata = f"{title} {description}"
    if TITLE_REJECTION[label].search(title) is not None:
        return None
    if HARD_NEGATIVE_TITLE[label].search(combined_metadata) is not None:
        return None
    if WEAK_POSITIVE_METADATA[label].search(combined_metadata) is not None:
        return None
    return AporeeCandidate(
        identifier=identifier,
        label=label,
        title=title,
        description=description,
        creator=creator,
        license=license_url,
        location_group=_location_group(identifier),
    )


def hard_negative_from_document(
    query_label: str, document: dict[str, object]
) -> AporeeCandidate | None:
    identifier = str(document.get("identifier", ""))
    title = str(document.get("title", ""))
    license_url = str(document.get("licenseurl", ""))
    if not identifier.startswith("aporee_") or not is_commercial_license(license_url):
        return None
    if TITLE_EVIDENCE[query_label].search(title) is None:
        return None
    creator_value = document.get("creator", "Radio Aporee contributor")
    if isinstance(creator_value, list):
        creator = ", ".join(str(value) for value in creator_value)
    else:
        creator = str(creator_value)
    description_value = document.get("description", "")
    if isinstance(description_value, list):
        description = " ".join(str(value) for value in description_value)
    else:
        description = str(description_value)
    if HARD_NEGATIVE_TITLE[query_label].search(f"{title} {description}") is None:
        return None
    return AporeeCandidate(
        identifier=identifier,
        label="other",
        title=title,
        description=description,
        creator=creator,
        license=license_url,
        location_group=_location_group(identifier),
    )


def _fetch_query(label: str) -> list[AporeeCandidate]:
    response = _request_json(
        ARCHIVE_SEARCH_URL,
        params={
            "q": f"collection:radio-aporee-maps AND {APOREE_QUERIES[label]}",
            "fl[]": [
                "identifier",
                "title",
                "description",
                "creator",
                "licenseurl",
            ],
            "rows": 1000,
            "page": 1,
            "output": "json",
        },
    )
    assert isinstance(response, dict)
    documents = response["response"]["docs"]
    candidates = [
        candidate_from_document(label, row)
        or hard_negative_from_document(label, row)
        for row in documents
    ]
    return [candidate for candidate in candidates if candidate is not None]


def load_candidates(cache_path: Path, *, cache_only: bool) -> list[AporeeCandidate]:
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return [AporeeCandidate(**row) for row in payload["items"]]
    if cache_only:
        return []

    by_identifier: dict[str, list[AporeeCandidate]] = {}
    for label in APOREE_QUERIES:
        for candidate in _fetch_query(label):
            by_identifier.setdefault(candidate.identifier, []).append(candidate)

    # An item or map location that matches two product classes is not reliable
    # ground truth.  Exclude the complete location instead of guessing a label.
    unambiguous = {
        identifier: values[0]
        for identifier, values in by_identifier.items()
        if len({value.label for value in values}) == 1
    }
    labels_by_location: dict[str, set[str]] = {}
    for candidate in unambiguous.values():
        labels_by_location.setdefault(candidate.location_group, set()).add(candidate.label)
    result = sorted(
        (
            candidate for candidate in unambiguous.values()
            if len(labels_by_location[candidate.location_group]) == 1
        ),
        key=lambda candidate: (candidate.label, candidate.location_group, candidate.identifier),
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {"version": 1, "items": [asdict(candidate) for candidate in result]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return result


def _audio_file(metadata: dict[str, object]) -> str:
    files = metadata.get("files", [])
    assert isinstance(files, list)
    priorities = ("VBR MP3", "Ogg Vorbis", "Flac", "WAVE")
    for preferred_format in priorities:
        for row in files:
            if not isinstance(row, dict) or row.get("format") != preferred_format:
                continue
            name = str(row.get("name", ""))
            if name and not name.endswith(("_spectrogram.png", "_files.xml")):
                return name
    raise RuntimeError(f"Radio Aporee item has no supported audio: {metadata.get('metadata')}")


def has_minimum_audio(path: Path, minimum_seconds: float = 5.0) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with wave.open(str(path), "rb") as stream:
            duration = stream.getnframes() / max(1, stream.getframerate())
    except (OSError, EOFError, wave.Error):
        return False
    return duration >= minimum_seconds


def fetch_audio_window(
    candidate: AporeeCandidate,
    target: Path,
    *,
    sample_rate: int,
    seconds: int,
    cache_only: bool,
    start_seconds: int = 0,
    optional: bool = False,
) -> Path | None:
    if has_minimum_audio(target):
        return target
    if cache_only:
        return None
    response = _request_json(
        ARCHIVE_METADATA_URL.format(identifier=quote(candidate.identifier, safe=""))
    )
    assert isinstance(response, dict)
    filename = _audio_file(response)
    source_url = ARCHIVE_DOWNLOAD_URL.format(
        identifier=quote(candidate.identifier, safe=""),
        filename=quote(filename, safe=""),
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    command = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-i",
        source_url,
        "-ss",
        str(start_seconds),
        "-t",
        str(seconds),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        str(temporary),
    ]
    for attempt in range(4):
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode == 0 and temporary.exists() and temporary.stat().st_size > 0:
            if has_minimum_audio(temporary, min(5.0, float(seconds))):
                temporary.replace(target)
                return target
        temporary.unlink(missing_ok=True)
        if completed.returncode == 0 and optional:
            return None
        if attempt == 3:
            if optional:
                return None
            raise RuntimeError(
                f"unable to decode {candidate.identifier}: {completed.stderr[-500:]}"
            )
        time.sleep(2 ** attempt)
    return None
