import AsyncStorage from '@react-native-async-storage/async-storage';
import { disableNetwork, enableNetwork } from 'firebase/firestore';
import { db } from './firebase';

const FIRESTORE_QUOTA_COOLDOWN_KEY = 'firestore:quota_cooldown_until';
const DEFAULT_FIRESTORE_QUOTA_COOLDOWN_MS = 15 * 60 * 1000;

let restoredCooldown = false;
let restorePromise = null;
let firestoreQuotaCooldownUntil = 0;
let firestoreNetworkDisabled = false;

export function isFirestoreQuotaError(error) {
  const code = String(error?.code || '').toLowerCase();
  const message = String(error?.message || '').toLowerCase();
  return code.includes('resource-exhausted') || message.includes('quota exceeded');
}

async function setFirestoreNetworkDisabled(disabled) {
  if (!db) {
    return false;
  }

  if (disabled) {
    if (firestoreNetworkDisabled) {
      return true;
    }

    try {
      await disableNetwork(db);
      firestoreNetworkDisabled = true;
      return true;
    } catch {
      return false;
    }
  }

  if (!firestoreNetworkDisabled) {
    return true;
  }

  try {
    await enableNetwork(db);
    firestoreNetworkDisabled = false;
    return true;
  } catch {
    return false;
  }
}

async function restorePersistedCooldown() {
  if (restoredCooldown) {
    return firestoreQuotaCooldownUntil;
  }

  if (restorePromise) {
    return restorePromise;
  }

  restorePromise = (async () => {
    try {
      const rawValue = await AsyncStorage.getItem(FIRESTORE_QUOTA_COOLDOWN_KEY);
      firestoreQuotaCooldownUntil = Number(rawValue || 0) || 0;
    } catch {
      firestoreQuotaCooldownUntil = 0;
    }

    restoredCooldown = true;
    restorePromise = null;

    if (firestoreQuotaCooldownUntil > Date.now()) {
      await setFirestoreNetworkDisabled(true);
      return firestoreQuotaCooldownUntil;
    }

    firestoreQuotaCooldownUntil = 0;
    await AsyncStorage.removeItem(FIRESTORE_QUOTA_COOLDOWN_KEY).catch(() => {});
    await setFirestoreNetworkDisabled(false);
    return 0;
  })();

  return restorePromise;
}

export async function prepareFirestoreAccess() {
  await restorePersistedCooldown();

  if (firestoreQuotaCooldownUntil > Date.now()) {
    await setFirestoreNetworkDisabled(true);
    return false;
  }

  firestoreQuotaCooldownUntil = 0;
  await AsyncStorage.removeItem(FIRESTORE_QUOTA_COOLDOWN_KEY).catch(() => {});
  await setFirestoreNetworkDisabled(false);
  return true;
}

export async function enterFirestoreQuotaCooldown(
  durationMs = DEFAULT_FIRESTORE_QUOTA_COOLDOWN_MS,
) {
  firestoreQuotaCooldownUntil = Math.max(
    firestoreQuotaCooldownUntil,
    Date.now() + Math.max(1000, Number(durationMs) || DEFAULT_FIRESTORE_QUOTA_COOLDOWN_MS),
  );
  restoredCooldown = true;

  await AsyncStorage.setItem(
    FIRESTORE_QUOTA_COOLDOWN_KEY,
    String(firestoreQuotaCooldownUntil),
  ).catch(() => {});
  await setFirestoreNetworkDisabled(true);

  return firestoreQuotaCooldownUntil;
}
