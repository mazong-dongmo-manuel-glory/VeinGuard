import { create } from 'zustand';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as SecureStore from 'expo-secure-store';
import {
  auth,
  FIREBASE_ENABLED,
} from '../services/firebase';
import {
  createUserWithEmailAndPassword,
  onAuthStateChanged,
  signInWithEmailAndPassword,
  signOut,
} from 'firebase/auth';
import { getDefaultPreferences, loadUserPreferences, saveUserPreferences } from '../services/preferences';

const REMEMBER_KEY = 'auth_remember_session';
const EMAIL_KEY = 'auth_email';
const PASSWORD_KEY = 'auth_password';

const normalizeEmail = (value) => String(value || '').trim().toLowerCase();

export const useAuthStore = create((set, get) => ({
  user: null,
  authReady: false,
  rememberSession: false,
  preferences: getDefaultPreferences(),

  bootstrap: async () => {
    if (!FIREBASE_ENABLED || !auth) {
      set({ authReady: true });
      return;
    }

    const rememberSession = (await AsyncStorage.getItem(REMEMBER_KEY)) === '1';
    set({ rememberSession });

    let resolved = false;
    const unsubscribe = onAuthStateChanged(auth, async (user) => {
      const preferences = user ? await loadUserPreferences(user.uid) : getDefaultPreferences();
      set({
        user,
        authReady: true,
        preferences,
      });
      resolved = true;
      unsubscribe();
    });

    const currentUser = auth.currentUser;
    if (!currentUser && rememberSession) {
      const email = await SecureStore.getItemAsync(EMAIL_KEY);
      const password = await SecureStore.getItemAsync(PASSWORD_KEY);
      if (email && password) {
        try {
          await signInWithEmailAndPassword(auth, email, password);
        } catch {
          await AsyncStorage.removeItem(REMEMBER_KEY);
          await SecureStore.deleteItemAsync(EMAIL_KEY);
          await SecureStore.deleteItemAsync(PASSWORD_KEY);
        }
      }
    }

    setTimeout(async () => {
      if (!resolved) {
        const fallbackUser = auth.currentUser;
        const preferences = fallbackUser ? await loadUserPreferences(fallbackUser.uid) : getDefaultPreferences();
        set({ user: fallbackUser, authReady: true, preferences });
      }
    }, 1200);
  },

  login: async ({ email, password, rememberSession }) => {
    if (!FIREBASE_ENABLED || !auth) {
      throw new Error('Firebase Authentication is not available.');
    }

    const normalizedEmail = normalizeEmail(email);
    const normalizedPassword = String(password || '');
    const credential = await signInWithEmailAndPassword(auth, normalizedEmail, normalizedPassword);

    if (rememberSession) {
      await AsyncStorage.setItem(REMEMBER_KEY, '1');
      await SecureStore.setItemAsync(EMAIL_KEY, normalizedEmail);
      await SecureStore.setItemAsync(PASSWORD_KEY, normalizedPassword);
    } else {
      await AsyncStorage.removeItem(REMEMBER_KEY);
      await SecureStore.deleteItemAsync(EMAIL_KEY);
      await SecureStore.deleteItemAsync(PASSWORD_KEY);
    }

    const preferences = await loadUserPreferences(credential.user.uid);
    set({
      user: credential.user,
      rememberSession: !!rememberSession,
      preferences,
    });
    return credential.user;
  },

  signup: async ({ email, password, rememberSession }) => {
    if (!FIREBASE_ENABLED || !auth) {
      throw new Error('Firebase Authentication is not available.');
    }

    const normalizedEmail = normalizeEmail(email);
    const normalizedPassword = String(password || '');
    const credential = await createUserWithEmailAndPassword(auth, normalizedEmail, normalizedPassword);
    await saveUserPreferences(credential.user.uid, getDefaultPreferences());
    await get().login({ email: normalizedEmail, password: normalizedPassword, rememberSession });
    return credential.user;
  },

  logout: async () => {
    if (auth) {
      await signOut(auth);
    }
    await AsyncStorage.removeItem(REMEMBER_KEY);
    await SecureStore.deleteItemAsync(EMAIL_KEY);
    await SecureStore.deleteItemAsync(PASSWORD_KEY);
    set({
      user: null,
      rememberSession: false,
      preferences: getDefaultPreferences(),
    });
  },

  updatePreferences: async (partialPreferences) => {
    const user = get().user;
    if (!user) {
      return get().preferences;
    }

    const nextPreferences = {
      ...get().preferences,
      ...partialPreferences,
    };
    const saved = await saveUserPreferences(user.uid, nextPreferences);
    set({ preferences: saved });
    return saved;
  },
}));
