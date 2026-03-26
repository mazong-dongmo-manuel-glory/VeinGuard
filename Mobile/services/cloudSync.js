import { collection, deleteDoc, doc, getDocs, serverTimestamp, setDoc } from 'firebase/firestore';
import { db, FIREBASE_ENABLED } from './firebase';

async function writeDocument(collectionName, documentId, payload) {
  if (!FIREBASE_ENABLED || !db || !documentId) {
    return false;
  }

  await setDoc(
    doc(db, collectionName, String(documentId)),
    {
      ...payload,
      updated_at: serverTimestamp(),
    },
    { merge: true },
  );
  return true;
}

export async function syncUserProfile(userId, payload) {
  return writeDocument('users', userId, {
    created_at: payload?.created_at || new Date().toISOString(),
    ...payload,
  });
}

export async function syncBiometricProfile(userId, payload) {
  return writeDocument('biometric_profiles', userId, payload);
}

export async function deleteUserProfile(userId) {
  if (!FIREBASE_ENABLED || !db || !userId) {
    return false;
  }
  await deleteDoc(doc(db, 'users', String(userId)));
  await deleteDoc(doc(db, 'biometric_profiles', String(userId)));
  return true;
}

export async function syncAccessEvent(eventId, payload) {
  return writeDocument('access_events', eventId, payload);
}

export async function syncTelemetry(deviceId, payload) {
  return writeDocument('device_telemetry', deviceId, payload);
}

function normalizeFirestoreValue(value) {
  if (value && typeof value?.toDate === 'function') {
    return value.toDate().toISOString();
  }
  return value;
}

export async function loadUserProfiles() {
  if (!FIREBASE_ENABLED || !db) {
    return [];
  }

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

  return users;
}
