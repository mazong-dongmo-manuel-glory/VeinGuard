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
import { useMqttStore } from '../store/mqttStore';

export default function Login({ navigation }) {
  const { t } = useTranslation();
  const { width, height } = useWindowDimensions();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  
  const login = useMqttStore((state) => state.login);
  const isConnected = useMqttStore((state) => state.isConnected);

  const handleLogin = async () => {
    if (!username || !password) {
      Alert.alert("Erreur", "Veuillez entrer un nom d'utilisateur et un mot de passe.");
      return;
    }

    if (!isConnected) {
      Alert.alert("Erreur Connexion", "Le système est hors-ligne. Veuillez vérifier le broker MQTT.");
      return;
    }

    try {
      const response = await login(username, password);

      if (response && response.status === 'success') {
        if (navigation) navigation.navigate('Dashboard');
      } else {
        Alert.alert("Échec de connexion", response.error || "Identifiants incorrects");
      }
    } catch (error) {
      Alert.alert("Erreur Système", "Le serveur de sécurité ne répond pas.");
      console.error(error);
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
                <Text style={styles.logoVein}>VEIN</Text>
                <Text style={styles.logoGuard}>GUARD</Text>
              </Text>
              <Text style={styles.logoSubtitle}>{t('login.subtitle').toUpperCase()}</Text>
            </View>

            {/* Glassmorphic Login Card */}
            <BlurView intensity={20} tint="dark" style={styles.glassCard}>
              <View style={styles.cardHeader}>
                <Text style={styles.cardTitle}>{t('login.accessSystem')}</Text>
                <View style={styles.statusBadge}>
                  <View style={styles.statusDot} />
                  <Text style={styles.statusText}>{t('login.mqttBadge')}</Text>
                </View>
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
                    value={username}
                    onChangeText={setUsername}
                    autoCapitalize="none"
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

              <TouchableOpacity style={styles.forgotBtn}>
                <Text style={styles.forgotText}>{t('login.forgotPassword')}</Text>
              </TouchableOpacity>

              {/* MAIN ACTION */}
              <TouchableOpacity style={styles.primaryBtn} onPress={handleLogin} activeOpacity={0.8}>
                <LinearGradient
                  colors={GRADIENTS.neonCyan}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 0 }}
                  style={styles.btnGradient}
                >
                  <Text style={styles.btnText}>{t('login.accessSystem').toUpperCase()}</Text>
                  <Ionicons name="arrow-forward" size={18} color={COLORS.white} />
                </LinearGradient>
              </TouchableOpacity>
            </BlurView>

            {/* Quick Access Divider */}
            <View style={styles.divider}>
              <View style={styles.line} />
              <Text style={styles.dividerText}>{t('login.orQuickAccess').toUpperCase()}</Text>
              <View style={styles.line} />
            </View>

            {/* Alternative Methods */}
            <View style={styles.quickActions}>
              <TouchableOpacity style={[styles.secondaryBtn, { borderColor: COLORS.neonGreen }]}>
                <Ionicons name="finger-print" size={22} color={COLORS.neonGreen} />
                <Text style={[styles.secondaryBtnText, { color: COLORS.neonGreen }]}>{t('login.biometricLogin')}</Text>
              </TouchableOpacity>

              <TouchableOpacity style={[styles.secondaryBtn, { borderColor: COLORS.neonMagenta }]}>
                <Ionicons name="scan-outline" size={22} color={COLORS.neonMagenta} />
                <Text style={[styles.secondaryBtnText, { color: COLORS.neonMagenta }]}>{t('login.veinBtn')}</Text>
              </TouchableOpacity>
            </View>

            <View style={styles.footer}>
              <Text style={styles.footerText}>SECURE ACCESS PROTOCOL v2.4</Text>
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
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 25,
  },
  cardTitle: { color: COLORS.white, fontSize: 18, fontWeight: '800' },
  statusBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(57, 255, 20, 0.1)',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(57, 255, 20, 0.3)',
  },
  statusDot: { width: 6, height: 6, borderRadius: 3, backgroundColor: COLORS.neonGreen, marginRight: 6 },
  statusText: { color: COLORS.neonGreen, fontSize: 10, fontWeight: '700' },

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

  divider: { flexDirection: 'row', alignItems: 'center', marginVertical: 35, width: '100%' },
  line: { flex: 1, height: 1, backgroundColor: 'rgba(255, 255, 255, 0.1)' },
  dividerText: { color: COLORS.textDim, fontSize: 10, marginHorizontal: 15, letterSpacing: 2, fontWeight: '700' },

  quickActions: { gap: 15, width: '100%' },
  secondaryBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 16,
    borderRadius: 16,
    borderWidth: 1.5,
    backgroundColor: 'transparent',
    gap: 12,
  },
  secondaryBtnText: { fontSize: 14, fontWeight: '800', letterSpacing: 1 },

  footer: { marginTop: 40, alignItems: 'center' },
  footerText: { color: COLORS.textDim, fontSize: 10, letterSpacing: 2, marginBottom: 15 },
  footerOrbs: { flexDirection: 'row', gap: 8 },
  miniOrb: { width: 6, height: 6, borderRadius: 3, opacity: 0.5 },
});
