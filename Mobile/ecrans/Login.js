import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  ScrollView,
  StatusBar,
  KeyboardAvoidingView,
  Platform,
  useWindowDimensions,
  Alert,
} from 'react-native';
import { useTranslation } from 'react-i18next';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { BlurView } from 'expo-blur';
import { COLORS, GRADIENTS, SHADOWS } from '../theme';
import { getFirebaseAuthErrorMessage, requestPasswordReset } from '../services/auth';
import { useAuthStore } from '../store/authStore';

export default function Login({ navigation }) {
  const { t } = useTranslation();
  const { width } = useWindowDimensions();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [rememberSession, setRememberSession] = useState(true);
  const [isSignupMode, setIsSignupMode] = useState(false);
  const login = useAuthStore((state) => state.login);
  const signup = useAuthStore((state) => state.signup);

  const handleLogin = async () => {
    if (!email || !password) {
      Alert.alert("Erreur", "Veuillez entrer une adresse e-mail et un mot de passe.");
      return;
    }

    if (!String(email).includes('@')) {
      Alert.alert("Erreur", "Veuillez entrer une adresse e-mail valide.");
      return;
    }

    if (password.length < 6) {
      Alert.alert("Erreur", "Le mot de passe doit contenir au moins 6 caractères.");
      return;
    }

    if (isSignupMode && password !== confirmPassword) {
      Alert.alert("Erreur", "Les mots de passe ne correspondent pas.");
      return;
    }

    try {
      if (isSignupMode) {
        await signup({ email, password, rememberSession });
      } else {
        await login({ email, password, rememberSession });
      }
    } catch (error) {
      Alert.alert(isSignupMode ? "Échec de création du compte" : "Échec de connexion", getFirebaseAuthErrorMessage(error));
      console.error(error);
    }
  };

  const handleForgotPassword = async () => {
    try {
      await requestPasswordReset(email);
      Alert.alert(t('common.success'), t('login.resetPasswordDesc'));
    } catch (error) {
      Alert.alert(t('common.error'), getFirebaseAuthErrorMessage(error));
    }
  };

  const viewportWidth = Math.min(width, 520);
  const uiScale = Math.max(0.9, Math.min(viewportWidth / 390, 1.1));

  return (
    <View style={styles.screen}>
      <StatusBar barStyle="light-content" />
      <LinearGradient
        colors={GRADIENTS.primary}
        style={StyleSheet.absoluteFill}
      />
      
      {/* Decorative Blur Orbs */}
      <View style={[styles.orb, { top: -50, left: -50, backgroundColor: COLORS.neonPurple }]} />
      <View style={[styles.orb, { bottom: -100, right: -100, backgroundColor: COLORS.neonCyan }]} />

      <KeyboardAvoidingView
        style={styles.keyboardWrap}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      >
        <ScrollView
          contentContainerStyle={styles.scroll}
          showsVerticalScrollIndicator={false}
          keyboardShouldPersistTaps="handled"
        >
          <View style={[styles.content, { width: viewportWidth * 0.9 }]}>
            {/* Logo Unit */}
            <View style={styles.logoContainer}>
              <View style={[styles.logoGlow, SHADOWS.cyan]}>
                <LinearGradient
                  colors={GRADIENTS.neonCyan}
                  style={styles.logoIconCircle}
                >
                  <Ionicons name="scan" size={32 * uiScale} color={COLORS.white} />
                </LinearGradient>
              </View>
              <Text style={styles.logoTitle}>
                <Text style={styles.logoVein}>BIO</Text>
                <Text style={styles.logoGuard}>GUARD</Text>
              </Text>
              <Text style={styles.logoSubtitle}>{t('login.subtitle').toUpperCase()}</Text>
            </View>

            {/* Glassmorphic Login Card */}
            <BlurView intensity={20} tint="dark" style={styles.glassCard}>
              <View style={styles.cardHeader}>
                <Text style={styles.cardTitle}>{t('login.accessSystem')}</Text>
              </View>

              {/* INPUT FIELDS */}
              <View style={styles.inputGroup}>
                <Text style={styles.label}>{t('login.emailLabel')}</Text>
                <View style={styles.inputWrapper}>
                  <Ionicons name="person-outline" size={20} color={COLORS.neonCyan} style={styles.inputIcon} />
                  <TextInput
                    style={styles.input}
                    placeholder={t('login.emailPlaceholder')}
                    placeholderTextColor={COLORS.textDim}
                    value={email}
                    onChangeText={setEmail}
                    autoCapitalize="none"
                    keyboardType="email-address"
                  />
                </View>
              </View>

              <View style={styles.inputGroup}>
                <Text style={styles.label}>{t('login.passwordLabel')}</Text>
                <View style={styles.inputWrapper}>
                  <Ionicons name="lock-closed-outline" size={20} color={COLORS.neonCyan} style={styles.inputIcon} />
                  <TextInput
                    style={styles.input}
                    placeholder={t('login.passwordPlaceholder')}
                    placeholderTextColor={COLORS.textDim}
                    value={password}
                    onChangeText={setPassword}
                    secureTextEntry={!showPassword}
                  />
                  <TouchableOpacity onPress={() => setShowPassword(!showPassword)}>
                    <Ionicons name={showPassword ? "eye-off-outline" : "eye-outline"} size={20} color={COLORS.textDim} />
                  </TouchableOpacity>
                </View>
              </View>

              {isSignupMode && (
                <View style={styles.inputGroup}>
                  <Text style={styles.label}>{t('login.confirmPasswordLabel')}</Text>
                  <View style={styles.inputWrapper}>
                    <Ionicons name="shield-checkmark-outline" size={20} color={COLORS.neonCyan} style={styles.inputIcon} />
                    <TextInput
                      style={styles.input}
                      placeholder={t('login.confirmPasswordPlaceholder')}
                      placeholderTextColor={COLORS.textDim}
                      value={confirmPassword}
                      onChangeText={setConfirmPassword}
                      secureTextEntry={!showPassword}
                    />
                  </View>
                </View>
              )}

              <TouchableOpacity style={styles.rememberRow} onPress={() => setRememberSession((value) => !value)} activeOpacity={0.8}>
                <Ionicons
                  name={rememberSession ? 'checkbox' : 'square-outline'}
                  size={20}
                  color={rememberSession ? COLORS.neonCyan : COLORS.textDim}
                />
                <Text style={styles.rememberText}>{t('login.rememberSession')}</Text>
              </TouchableOpacity>

              {!isSignupMode && (
                <TouchableOpacity style={styles.forgotBtn} onPress={handleForgotPassword}>
                  <Text style={styles.forgotText}>{t('login.forgotPassword')}</Text>
                </TouchableOpacity>
              )}

              {/* MAIN ACTION */}
              <TouchableOpacity style={styles.primaryBtn} onPress={handleLogin} activeOpacity={0.8}>
                <LinearGradient
                  colors={GRADIENTS.neonCyan}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 0 }}
                  style={styles.btnGradient}
                >
                  <Text style={styles.btnText}>{isSignupMode ? t('login.createAccount') : t('login.accessSystem').toUpperCase()}</Text>
                  <Ionicons name="arrow-forward" size={18} color={COLORS.white} />
                </LinearGradient>
              </TouchableOpacity>

              <TouchableOpacity style={styles.authToggleBtn} onPress={() => setIsSignupMode((value) => !value)}>
                <Text style={styles.authToggleText}>
                  {isSignupMode ? t('login.loginInstead') : t('login.signupInstead')}
                </Text>
              </TouchableOpacity>
            </BlurView>

            <View style={styles.footer}>
              <Text style={styles.footerText}>{t('login.footer')}</Text>
              <View style={styles.footerOrbs}>
                <View style={[styles.miniOrb, { backgroundColor: COLORS.neonCyan }]} />
                <View style={[styles.miniOrb, { backgroundColor: COLORS.neonPurple }]} />
                <View style={[styles.miniOrb, { backgroundColor: COLORS.neonGreen }]} />
              </View>
            </View>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: COLORS.bg },
  keyboardWrap: { flex: 1 },
  scroll: { flexGrow: 1, justifyContent: 'center', alignItems: 'center', paddingVertical: 40 },
  content: { alignItems: 'center' },
  
  orb: {
    position: 'absolute',
    width: 300,
    height: 300,
    borderRadius: 150,
    opacity: 0.15,
    blurRadius: 100,
  },

  logoContainer: { alignItems: 'center', marginBottom: 40 },
  logoGlow: { marginBottom: 20 },
  logoIconCircle: {
    width: 80,
    height: 80,
    borderRadius: 40,
    justifyContent: 'center',
    alignItems: 'center',
  },
  logoTitle: { fontSize: 32, fontWeight: '900', letterSpacing: 4, marginBottom: 5 },
  logoVein: { color: COLORS.white },
  logoGuard: { color: COLORS.neonCyan },
  logoSubtitle: { color: COLORS.textSecondary, fontSize: 10, letterSpacing: 6 },

  glassCard: {
    width: '100%',
    padding: 24,
    borderRadius: 30,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
    overflow: 'hidden',
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'flex-start',
    alignItems: 'center',
    marginBottom: 25,
  },
  cardTitle: { color: COLORS.white, fontSize: 18, fontWeight: '800' },

  inputGroup: { marginBottom: 20 },
  label: { color: COLORS.textSecondary, fontSize: 12, fontWeight: '600', marginBottom: 10, marginLeft: 5 },
  inputWrapper: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: 16,
    paddingHorizontal: 15,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  inputIcon: { marginRight: 12 },
  input: { flex: 1, paddingVertical: 15, color: COLORS.white, fontSize: 15 },

  rememberRow: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 18 },
  rememberText: { color: COLORS.textSecondary, fontSize: 13, fontWeight: '500' },
  forgotBtn: { alignSelf: 'flex-end', marginBottom: 25 },
  forgotText: { color: COLORS.neonCyan, fontSize: 13, fontWeight: '500' },

  primaryBtn: { borderRadius: 16, overflow: 'hidden' },
  btnGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 18,
    gap: 10,
  },
  btnText: { color: COLORS.white, fontSize: 15, fontWeight: '900', letterSpacing: 2 },
  authToggleBtn: { marginTop: 16, alignSelf: 'center' },
  authToggleText: { color: COLORS.neonCyan, fontSize: 13, fontWeight: '600' },

  footer: { marginTop: 40, alignItems: 'center' },
  footerText: { color: COLORS.textDim, fontSize: 10, letterSpacing: 2, marginBottom: 15 },
  footerOrbs: { flexDirection: 'row', gap: 8 },
  miniOrb: { width: 6, height: 6, borderRadius: 3, opacity: 0.5 },
});
