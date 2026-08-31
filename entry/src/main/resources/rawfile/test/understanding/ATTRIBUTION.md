# Audio understanding evaluation data

This directory contains a 30-file evaluation subset. Audio is converted to
16 kHz, mono, PCM16 WAV for deterministic on-device decoding.

## CREMA-D

- Source: https://github.com/CheyneyComputerScience/CREMA-D
- License: Open Database License 1.0; individual contents under the Database
  Contents License 1.0.
- Included: 18 clips, three for each of anger, disgust, fear, happy, neutral,
  and sad.
- Selection: the acted filename label must match `VoiceVote` in the official
  `processedResults/summaryTable.csv`; the highest-agreement clips were
  selected while keeping speaker variety.
- Citation: Cao et al., "CREMA-D: Crowd-sourced Emotional Multimodal Actors
  Dataset", IEEE Transactions on Affective Computing, 2014.

## VocalSound

- Source: https://github.com/YuanGongND/vocalsound
- License: Creative Commons Attribution-ShareAlike 4.0 International.
- Included: 12 official test-split clips: four cough, four laughter, and four
  sneeze examples from different speakers.
- Selection: only event classes with an exact SenseVoice output label are
  included in the scored protocol.
- Citation: Gong, Yu, and Glass, "VocalSound: A Dataset for Improving Human
  Vocal Sounds Recognition", ICASSP 2022.

The subset is redistributed under the source licenses above. Model predictions
are not part of the dataset and are generated locally on the device.
