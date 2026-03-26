import AsyncStorage from '@react-native-async-storage/async-storage';
import { db, FIREBASE_ENABLED } from './firebase';
import { doc, getDoc, setDoc } from 'firebase/firestore';

const DEFAULT_PREFERENCES = {
  autoRefreshData: true,
  showTechnicalDetails: true,
  compactLists: false,
};

const storageKey = (uid) => `user-preferences:${uid}`;

export function getDefaultPreferences() {
  return { ...DEFAULT_PREFERENCES };
}

export async function loadUserPreferences(uid) {
  if (!uid) {
    return getDefaultPreferences();
  }

  const cached = await AsyncStorage.getItem(storageKey(uid));
  if (cached) {
    try {
      return { ...DEFAULT_PREFERENCES, ...JSON.parse(cached) };
    } catch {}
  }

  if (FIREBASE_ENABLED && db) {
    try {
      const snapshot = await getDoc(doc(db, 'mobile_user_preferences', uid));
      if (snapshot.exists()) {
        const prefs = { ...DEFAULT_PREFERENCES, ...snapshot.data() };
        await AsyncStorage.setItem(storageKey(uid), JSON.stringify(prefs));
        return prefs;
      }
    } catch {}
  }

  return getDefaultPreferences();
}

export async function saveUserPreferences(uid, preferences) {
  if (!uid) {
    return getDefaultPreferences();
  }

  const payload = { ...DEFAULT_PREFERENCES, ...preferences };
  await AsyncStorage.setItem(storageKey(uid), JSON.stringify(payload));

  if (FIREBASE_ENABLED && db) {
    try {
      await setDoc(doc(db, 'mobile_user_preferences', uid), payload, { merge: true });
    } catch {}
  }

  return payload;
}

export async function clearUserPreferences(uid) {
  if (!uid) return;
  await AsyncStorage.removeItem(storageKey(uid));
}
