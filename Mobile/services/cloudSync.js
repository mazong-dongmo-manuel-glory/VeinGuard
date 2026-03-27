import AsyncStorage from '@react-native-async-storage/async-storage';
import { collection, deleteDoc, doc, getDocs, serverTimestamp, setDoc } from 'firebase/firestore';
import { db, FIRESTORE_ENABLED } from './firebase';
import {
  enterFirestoreQuotaCooldown,
  isFirestoreQuotaError,
  prepareFirestoreAccess,
} from './firestoreGuard';

const USER_PROFILES_CACHE_KEY = 'cloud_sync:user_profiles';
const USER_PROFILES_CACHE_TTL_MS = 5 * 60 * 1000;
const TELEMETRY_MIN_SYNC_INTERVAL_MS = 60 * 1000;

let lastTelemetrySyncAt = 0;

function isObject(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function stripBase64Fields(value) {
  if (Array.isArray(value)) {
    return value.map(stripBase64Fields);
  }

  if (!isObject(value)) {
    return value;
  }

  return Object.fromEntries(
    Object.entries(value)
      .filter(([key]) => !String(key).toLowerCase().includes('base64'))
      .map(([key, nestedValue]) => [key, stripBase64Fields(nestedValue)]),
  );
}

function sanitizePayload(payload) {
  return stripBase64Fields(payload || {});
}

async function readCachedUserProfiles() {
  try {
    const raw = await AsyncStorage.getItem(USER_PROFILES_CACHE_KEY);
    if (!raw) {
      return null;
    }

    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed?.users)) {
      return null;
    }

    return parsed;
  } catch {
    return null;
  }
}

async function writeCachedUserProfiles(users) {
  try {
    await AsyncStorage.setItem(
      USER_PROFILES_CACHE_KEY,
      JSON.stringify({
        cachedAt: Date.now(),
        users,
      }),
    );
  } catch {}
}

async function upsertCachedUserProfile(userId, payload) {
  const current = (await readCachedUserProfiles())?.users || [];
  const nextEntry = {
    id: String(userId),
    ...sanitizePayload(payload),
  };
  const nextUsers = current.filter((entry) => String(entry?.id) !== String(userId));
  nextUsers.push(nextEntry);
  nextUsers.sort((left, right) => {
    const a = String(right.created_at || right.updated_at || '');
    const b = String(left.created_at || left.updated_at || '');
    return a.localeCompare(b);
  });
  await writeCachedUserProfiles(nextUsers);
}

async function removeCachedUserProfile(userId) {
  const current = (await readCachedUserProfiles())?.users || [];
  const nextUsers = current.filter((entry) => String(entry?.id) !== String(userId));
  await writeCachedUserProfiles(nextUsers);
}

function isCacheFresh(cache) {
  return Boolean(cache?.cachedAt) && Date.now() - Number(cache.cachedAt) <= USER_PROFILES_CACHE_TTL_MS;
}

async function writeDocument(collectionName, documentId, payload) {
  if (!FIRESTORE_ENABLED || !db || !documentId || !(await prepareFirestoreAccess())) {
    return false;
  }

  try {
    await setDoc(
      doc(db, collectionName, String(documentId)),
      {
        ...sanitizePayload(payload),
        updated_at: serverTimestamp(),
      },
      { merge: true },
    );
    return true;
  } catch (error) {
    if (isFirestoreQuotaError(error)) {
      await enterFirestoreQuotaCooldown();
      return false;
    }
    throw error;
  }
}

export async function syncUserProfile(userId, payload) {
  if (!userId) {
    return false;
  }

  const nextPayload = {
    created_at: payload?.created_at || new Date().toISOString(),
    ...payload,
  };
  await upsertCachedUserProfile(userId, nextPayload);
  return writeDocument('users', userId, nextPayload);
}

export async function syncBiometricProfile(userId, payload) {
  return writeDocument('biometric_profiles', userId, payload);
}

export async function deleteUserProfile(userId) {
  if (!userId) {
    return false;
  }

  if (!FIRESTORE_ENABLED || !db || !(await prepareFirestoreAccess())) {
    await removeCachedUserProfile(userId);
    return false;
  }

  try {
    await deleteDoc(doc(db, 'users', String(userId)));
    await deleteDoc(doc(db, 'biometric_profiles', String(userId)));
    await removeCachedUserProfile(userId);
    return true;
  } catch (error) {
    if (isFirestoreQuotaError(error)) {
      await enterFirestoreQuotaCooldown();
      await removeCachedUserProfile(userId);
      return false;
    }
    throw error;
  }
}

export async function syncAccessEvent(eventId, payload) {
  return writeDocument('access_events', eventId, payload);
}

export async function syncTelemetry(deviceId, payload) {
  if (Date.now() - lastTelemetrySyncAt < TELEMETRY_MIN_SYNC_INTERVAL_MS) {
    return false;
  }

  lastTelemetrySyncAt = Date.now();
  return writeDocument('device_telemetry', deviceId, payload);
}

function normalizeFirestoreValue(value) {
  if (value && typeof value?.toDate === 'function') {
    return value.toDate().toISOString();
  }
  return value;
}

export async function loadUserProfiles() {
  const cached = await readCachedUserProfiles();
  const firestoreAvailable =
    FIRESTORE_ENABLED && db ? await prepareFirestoreAccess() : false;

  if ((!FIRESTORE_ENABLED || !db || !firestoreAvailable) && Array.isArray(cached?.users)) {
    return cached.users;
  }

  if (!FIRESTORE_ENABLED || !db) {
    return Array.isArray(cached?.users) ? cached.users : [];
  }

  if (isCacheFresh(cached)) {
    return cached.users;
  }

  try {
    const snapshot = await getDocs(collection(db, 'users'));
    const users = snapshot.docs.map((entry) => {
      const data = entry.data() || {};
      return {
        id: String(entry.id),
        ...Object.fromEntries(
          Object.entries(data).map(([key, value]) => [key, normalizeFirestoreValue(value)]),
        ),
      };
    });

    users.sort((left, right) => {
      const a = String(right.created_at || right.updated_at || '');
      const b = String(left.created_at || left.updated_at || '');
      return a.localeCompare(b);
    });

    await writeCachedUserProfiles(users);
    return users;
  } catch (error) {
    if (isFirestoreQuotaError(error)) {
      await enterFirestoreQuotaCooldown();
      return Array.isArray(cached?.users) ? cached.users : [];
    }
    throw error;
  }
}
