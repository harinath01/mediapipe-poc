declare global {
    interface Window {
      cv: typeof import('mirada/dist/src/types/opencv/_types');
    }
}