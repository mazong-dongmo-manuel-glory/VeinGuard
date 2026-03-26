import { initializeApp, getApps } from 'firebase/app';
import { getFirestore } from 'firebase/firestore';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Platform } from 'react-native';
import { getAuth } from 'firebase/auth';
import { getReactNativePersistence, initializeAuth } from 'firebase/auth/react-native';
import { FIREBASE_CONFIG, FIREBASE_ENABLED } from '../config';

let app = null;
let db = null;
let auth = null;

if (FIREBASE_ENABLED) {
  app = getApps().length ? getApps()[0] : initializeApp(FIREBASE_CONFIG);
  db = getFirestore(app);

  if (Platform.OS === 'web') {
    auth = getAuth(app);
  } else {
    try {
      auth = initializeAuth(app, {
        persistence: getReactNativePersistence(AsyncStorage),
      });
    } catch (error) {
      auth = getAuth(app);
    }
  }
}

export { app, auth, db, FIREBASE_ENABLED };
