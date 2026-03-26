import {
  createUserWithEmailAndPassword,
  sendPasswordResetEmail,
  signInWithEmailAndPassword,
  signOut,
} from 'firebase/auth';
import i18next from 'i18next';
import { auth, FIREBASE_ENABLED } from './firebase';

export async function loginWithEmailPassword(email, password) {
  if (!FIREBASE_ENABLED || !auth) {
    throw new Error(i18next.t('login.firebaseUnavailable'));
  }

  const normalizedEmail = String(email || '').trim().toLowerCase();
  const normalizedPassword = String(password || '');

  if (!normalizedEmail || !normalizedPassword) {
    throw new Error(i18next.t('login.missingCredentials'));
  }

  const credential = await signInWithEmailAndPassword(auth, normalizedEmail, normalizedPassword);
  return credential.user;
}

export function getFirebaseAuthErrorMessage(error) {
  const code = error?.code || '';

  switch (code) {
    case 'auth/invalid-email':
      return i18next.t('login.authInvalidEmail');
    case 'auth/invalid-credential':
      return i18next.t('login.authInvalidCredential');
    case 'auth/user-not-found':
      return i18next.t('login.authUserNotFound');
    case 'auth/wrong-password':
      return i18next.t('login.authWrongPassword');
    case 'auth/too-many-requests':
      return i18next.t('login.authTooManyRequests');
    case 'auth/network-request-failed':
      return i18next.t('login.authNetworkFailed');
    case 'auth/operation-not-allowed':
      return i18next.t('login.authOperationNotAllowed');
    case 'auth/email-already-in-use':
      return i18next.t('login.authEmailInUse');
    case 'auth/weak-password':
      return i18next.t('login.authWeakPassword');
    case 'auth/missing-email':
      return i18next.t('login.authMissingEmail');
    default:
      return error?.message || i18next.t('login.authDefaultError');
  }
}

export async function signupWithEmailPassword(email, password) {
  if (!FIREBASE_ENABLED || !auth) {
    throw new Error(i18next.t('login.firebaseUnavailable'));
  }

  const normalizedEmail = String(email || '').trim().toLowerCase();
  const normalizedPassword = String(password || '');

  if (!normalizedEmail || !normalizedPassword) {
    throw new Error(i18next.t('login.missingCredentials'));
  }

  const credential = await createUserWithEmailAndPassword(auth, normalizedEmail, normalizedPassword);
  return credential.user;
}

export async function logoutFromFirebase() {
  if (!FIREBASE_ENABLED || !auth) {
    return;
  }

  await signOut(auth);
}

export async function requestPasswordReset(email) {
  if (!FIREBASE_ENABLED || !auth) {
    throw new Error(i18next.t('login.firebaseUnavailable'));
  }

  const normalizedEmail = String(email || '').trim().toLowerCase();
  if (!normalizedEmail) {
    throw new Error(i18next.t('login.authMissingEmail'));
  }

  await sendPasswordResetEmail(auth, normalizedEmail);
}
