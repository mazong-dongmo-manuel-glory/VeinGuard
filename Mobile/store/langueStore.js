import AsyncStorage from "@react-native-async-storage/async-storage";
import i18next from "i18next";

const STORAGE_KEY = "langue-storage";

let state = {
  langue: "fr",
  modifierLangue: (nouvelleLangue) => {
    if (!nouvelleLangue) return;
    state = { ...state, langue: nouvelleLangue };
    i18next.changeLanguage(nouvelleLangue);
    AsyncStorage.setItem(STORAGE_KEY, nouvelleLangue).catch(() => {});
    listeners.forEach((listener) => listener());
  },
};

const listeners = new Set();

AsyncStorage.getItem(STORAGE_KEY)
  .then((savedLangue) => {
    if (savedLangue) {
      state = { ...state, langue: savedLangue };
      i18next.changeLanguage(savedLangue);
      listeners.forEach((listener) => listener());
    }
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
