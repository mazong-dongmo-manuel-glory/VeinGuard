import AsyncStorage from '@react-native-async-storage/async-storage';
import i18next from 'i18next';

const STORAGE_KEY = 'langue-storage';
const DEFAULT_LANGUAGE = 'fr';
const SUPPORTED_LANGUAGES = ['fr', 'en'];

let state = {
  langue: DEFAULT_LANGUAGE,
  modifierLangue: async (nextLangue) => {
    const langue = SUPPORTED_LANGUAGES.includes(nextLangue) ? nextLangue : DEFAULT_LANGUAGE;
    state = { ...state, langue };
    try {
      await AsyncStorage.setItem(STORAGE_KEY, langue);
    } catch {}
    try {
      await i18next.changeLanguage(langue);
    } catch {}
    listeners.forEach((listener) => listener());
  },
};

const listeners = new Set();

AsyncStorage.getItem(STORAGE_KEY)
  .then(async (storedLangue) => {
    const langue = SUPPORTED_LANGUAGES.includes(storedLangue) ? storedLangue : DEFAULT_LANGUAGE;
    state = { ...state, langue };
    try {
      await i18next.changeLanguage(langue);
    } catch {}
    listeners.forEach((listener) => listener());
  })
  .catch(() => {});

export const useLangueStore = (selector = (currentState) => currentState) => {
  const React = require('react');
  return React.useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => selector(state),
    () => selector(state),
  );
};
