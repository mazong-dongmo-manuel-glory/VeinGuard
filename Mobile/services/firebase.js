import { initializeApp, getApps } from 'firebase/app';
import { getFirestore } from 'firebase/firestore';
import { getAuth } from 'firebase/auth';
import { FIREBASE_CONFIG, FIREBASE_ENABLED, FIRESTORE_ENABLED } from '../config';

let app = null;
let db = null;
let auth = null;

if (FIREBASE_ENABLED || FIRESTORE_ENABLED) {
  app = getApps().length ? getApps()[0] : initializeApp(FIREBASE_CONFIG);
}

if (FIREBASE_ENABLED && app) {
  auth = getAuth(app);
}

if (FIRESTORE_ENABLED && app) {
  db = getFirestore(app);
}

export { app, auth, db, FIREBASE_ENABLED, FIRESTORE_ENABLED };
