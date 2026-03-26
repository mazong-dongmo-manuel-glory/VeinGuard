import AsyncStorage from "@react-native-async-storage/async-storage";
import i18next from "i18next";

const STORAGE_KEY = "langue-storage";
const DEFAULT_LANGUAGE = "fr";

let state = {
  langue: DEFAULT_LANGUAGE,
  modifierLangue: () => {
    state = { ...state, langue: DEFAULT_LANGUAGE };
    i18next.changeLanguage(DEFAULT_LANGUAGE);
    AsyncStorage.setItem(STORAGE_KEY, DEFAULT_LANGUAGE).catch(() => {});
    listeners.forEach((listener) => listener());
  },
};

const listeners = new Set();

AsyncStorage.getItem(STORAGE_KEY)
  .then(() => {
    state = { ...state, langue: DEFAULT_LANGUAGE };
    i18next.changeLanguage(DEFAULT_LANGUAGE);
    AsyncStorage.setItem(STORAGE_KEY, DEFAULT_LANGUAGE).catch(() => {});
    listeners.forEach((listener) => listener());
  })
  .catch(() => {});

export const useLangueStore = (selector = (s) => s) => {
  const React = require("react");
  return React.useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => selector(state),
    () => selector(state),
  );
};
