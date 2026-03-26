import { signInWithEmailAndPassword, signOut } from 'firebase/auth';
import { auth, FIREBASE_ENABLED } from './firebase';

export async function loginWithEmailPassword(email, password) {
  if (!FIREBASE_ENABLED || !auth) {
    throw new Error('Firebase Authentication is not available.');
  }

  const credential = await signInWithEmailAndPassword(auth, email, password);
  return credential.user;
}

export async function logoutFromFirebase() {
  if (!FIREBASE_ENABLED || !auth) {
    return;
  }

  await signOut(auth);
}
