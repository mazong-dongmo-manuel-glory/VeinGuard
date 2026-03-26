import { deleteDoc, doc, serverTimestamp, setDoc } from 'firebase/firestore';
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
  return writeDocument('users', userId, payload);
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
