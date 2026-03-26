import { signInWithEmailAndPassword, signOut } from 'firebase/auth';
import { auth, FIREBASE_ENABLED } from './firebase';

export async function loginWithEmailPassword(email, password) {
  if (!FIREBASE_ENABLED || !auth) {
    throw new Error('Firebase Authentication is not available.');
  }

  const normalizedEmail = String(email || '').trim().toLowerCase();
  const normalizedPassword = String(password || '');

  if (!normalizedEmail || !normalizedPassword) {
    throw new Error('Veuillez entrer une adresse e-mail et un mot de passe.');
  }

  const credential = await signInWithEmailAndPassword(auth, normalizedEmail, normalizedPassword);
  return credential.user;
}

export function getFirebaseAuthErrorMessage(error) {
  const code = error?.code || '';

  switch (code) {
    case 'auth/invalid-email':
      return "L'adresse e-mail n'est pas valide.";
    case 'auth/invalid-credential':
      return "L'adresse e-mail ou le mot de passe est incorrect.";
    case 'auth/user-not-found':
      return "Aucun compte n'existe avec cette adresse e-mail.";
    case 'auth/wrong-password':
      return 'Le mot de passe est incorrect.';
    case 'auth/too-many-requests':
      return 'Trop de tentatives. Réessayez plus tard.';
    case 'auth/network-request-failed':
      return 'La connexion réseau a échoué.';
    case 'auth/operation-not-allowed':
      return "La connexion par e-mail et mot de passe n'est pas activée dans Firebase.";
    default:
      return error?.message || 'Échec de connexion.';
  }
}

export async function logoutFromFirebase() {
  if (!FIREBASE_ENABLED || !auth) {
    return;
  }

  await signOut(auth);
}
