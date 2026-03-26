import { initializeApp, getApps } from 'firebase/app';
import { getFirestore } from 'firebase/firestore';
import { getAuth } from 'firebase/auth';
import { FIREBASE_CONFIG, FIREBASE_ENABLED } from '../config';

let app = null;
let db = null;
let auth = null;

if (FIREBASE_ENABLED) {
  app = getApps().length ? getApps()[0] : initializeApp(FIREBASE_CONFIG);
  db = getFirestore(app);
  auth = getAuth(app);
}

export { app, auth, db, FIREBASE_ENABLED };
