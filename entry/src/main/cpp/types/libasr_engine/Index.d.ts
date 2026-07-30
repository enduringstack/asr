export function initialize(modelDir?: string): number;
export function initializeFromFd(modelDir: string, fd: number, offset: number, length: number, vocabData?: ArrayBuffer): number;
export function processChunk(pcmData: ArrayBuffer): string;
export function processChunkAsync(pcmData: ArrayBuffer): Promise<string>;
export function finalize(): string;
export function reset(): void;
export function isInitialized(): boolean;
