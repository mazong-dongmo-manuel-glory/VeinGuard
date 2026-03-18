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
} from 'react-native';
import { useTranslation } from 'react-i18next';

const COLORS = {
  bg: '#080e1a',
  cardBg: '#0d1b2e',
  cardBorder: '#1e3a5f',
  green: '#00ff88',
  teal: '#00e5ff',
  magenta: '#ff00e5',
  amber: '#f5a623',
  text: '#b8cfe0',
  textDim: '#4a6a8a',
  white: '#ffffff',
  inputBg: '#091525',
  borderBlue: '#1a3a6a',
  gradientStart: '#00e5ff',
  gradientEnd: '#d000ff',
};

export default function Login({ navigation }) {
  const { t } = useTranslation();
  const { width, height } = useWindowDimensions();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [domain, setDomain] = useState('Primary');

  const viewportWidth = Math.min(width, 520);
  const isNarrow = viewportWidth < 360;
  const isCompact = viewportWidth < 340;
  const isShort = height < 760;
  const isVeryShort = height < 690;
  const isTablet = viewportWidth >= 768;

  // Keep the login screen on the same visual grid as the other screens.
  const uiScale = Math.max(isVeryShort ? 0.88 : 0.93, Math.min(viewportWidth / 390, 1.02));
  const horizontalPadding = 14;
  const topPadding = isVeryShort ? 10 : isShort ? 14 : 28;
  const bottomPadding = isVeryShort ? 56 : isShort ? 36 : 34;
  const contentMaxWidth = isTablet ? 430 : viewportWidth - horizontalPadding * 2;

  const logoSize = Math.round((isVeryShort ? 48 : 60) * uiScale);
  const logoFontSize = Math.round((isVeryShort ? 24 : 28) * uiScale);
  const subtitleSpacing = isCompact ? 3 : 4;
  const cardPadding = Math.round((isVeryShort ? 12 : 16) * uiScale);
  const inputVertical = Math.round((isVeryShort ? 9 : 11) * uiScale);
  const actionBtnVertical = Math.round((isVeryShort ? 10 : 13) * uiScale);
  const fieldLabelSize = Math.max(10, Math.round(11 * uiScale));
  const inputSize = Math.max(12, Math.round((isVeryShort ? 13 : 14) * uiScale));
  const primaryBtnTextSize = Math.max(13, Math.round(14 * uiScale));
  const secondaryBtnTextSize = Math.max(12, Math.round(13 * uiScale));
  const cardRadius = 10;
  const cornerSize = 20;
  const cornerOffset = 8;

  return (
    <View style={styles.screen}>
      <StatusBar barStyle="light-content" backgroundColor={COLORS.bg} />
      {/* Corner decorations */}
      <View style={[styles.corner, styles.cornerTL, { width: cornerSize, height: cornerSize, top: cornerOffset, left: cornerOffset }]} />
      <View style={[styles.corner, styles.cornerTR, { width: cornerSize, height: cornerSize, top: cornerOffset, right: cornerOffset }]} />
      <View style={[styles.corner, styles.cornerBL, { width: cornerSize, height: cornerSize, bottom: cornerOffset, left: cornerOffset }]} />
      <View style={[styles.corner, styles.cornerBR, { width: cornerSize, height: cornerSize, bottom: cornerOffset, right: cornerOffset }]} />

      <KeyboardAvoidingView
        style={styles.keyboardWrap}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      >
        <ScrollView
          contentContainerStyle={[
            styles.scroll,
            {
              paddingHorizontal: horizontalPadding,
              paddingTop: topPadding,
              paddingBottom: bottomPadding,
              minHeight: height,
            },
          ]}
          showsVerticalScrollIndicator={false}
          keyboardShouldPersistTaps="handled"
        >
          <View style={[styles.contentWrap, { maxWidth: contentMaxWidth }]}> 
          {/* Logo section */}
          <View style={[styles.logoSection, { marginBottom: isVeryShort ? 12 : isCompact ? 18 : 24 }]}>
            <View style={[styles.logoIconCircle, { width: logoSize, height: logoSize, borderRadius: logoSize / 2 }]}> 
            <Text style={[styles.logoIconText, { fontSize: Math.round((isVeryShort ? 20 : 24) * uiScale) }]}>〜</Text>
          </View>
          <Text style={[styles.logoText, { fontSize: logoFontSize, letterSpacing: isCompact ? 2 : 3 }]}>
            <Text style={styles.logoVein}>VEIN</Text>
            <Text style={styles.logoGuard}>GUARD</Text>
          </Text>
          <Text style={[styles.logoSubtitle, { letterSpacing: subtitleSpacing, fontSize: Math.max(9, Math.round((isVeryShort ? 10 : 11) * uiScale)), marginBottom: isVeryShort ? 10 : 14 }]}>BIOMETRIC ACCESS CONTROL</Text>
          <View style={styles.mqttBadge}>
            <View style={styles.mqttDot} />
            <Text style={[styles.mqttText, { fontSize: Math.max(10, Math.round(11 * uiScale)) }]}>MQTT ONLINE</Text>
          </View>
        </View>

        {/* Form card */}
        <View style={[styles.formCard, { padding: cardPadding, borderRadius: cardRadius, marginBottom: isVeryShort ? 10 : isShort ? 16 : 20 }]}> 
          {/* Left accent bar */}
          <View style={styles.accentBar} />

          {/* USERNAME */}
          <View style={styles.fieldGroup}>
            <Text style={[styles.fieldLabel, { fontSize: fieldLabelSize }]}>
                            <Text style={styles.fieldIcon}>👤 </Text>
                            {t('systemSettings.username')}
                          </Text>
            <TextInput
              style={[styles.input, { paddingVertical: inputVertical, fontSize: inputSize }]}
              placeholder="Enter username or email"
              placeholderTextColor={COLORS.textDim}
              value={username}
              onChangeText={setUsername}
              autoCapitalize="none"
              keyboardType="email-address"
            />
          </View>

          {/* PASSWORD */}
          <View style={styles.fieldGroup}>
            <Text style={[styles.fieldLabel, { fontSize: fieldLabelSize }]}>
              <Text style={styles.fieldIcon}>🔒 </Text>
              {t('systemSettings.password')}
            </Text>
            <View style={styles.passwordRow}>
              <TextInput
                style={[styles.input, styles.passwordInput, { paddingVertical: inputVertical, fontSize: inputSize }]}
                placeholder={t('systemSettings.enterPassword')}
                placeholderTextColor={COLORS.textDim}
                value={password}
                onChangeText={setPassword}
                secureTextEntry={!showPassword}
              />
              <TouchableOpacity style={styles.eyeBtn} onPress={() => setShowPassword(!showPassword)}>
                <Text style={styles.eyeIcon}>{showPassword ? '🙈' : '👁'}</Text>
              </TouchableOpacity>
            </View>
          </View>

          {/* ADMIN DOMAIN */}
          <View
            style={[
              styles.domainRow,
              {
                borderRadius: Math.round(8 * uiScale),
              },
              isNarrow && styles.domainRowCompact,
            ]}
          >
            <View style={styles.domainLeft}>
              <Text style={styles.domainIcon}>🛡</Text>
              <Text style={[styles.domainLabel, { fontSize: fieldLabelSize }]}>ADMIN DOMAIN</Text>
            </View>
            <TouchableOpacity style={[styles.domainDropdown, isNarrow && styles.domainDropdownCompact]}>
              <Text style={[styles.domainDropdownText, { fontSize: Math.max(11, Math.round(12 * uiScale)) }]}>{domain}  ▼</Text>
            </TouchableOpacity>
          </View>

          {/* FORGOT PASSWORD */}
          <TouchableOpacity style={styles.forgotRow}>
            <Text style={[styles.forgotText, { fontSize: Math.max(11, Math.round(12 * uiScale)) }]}>🔑 FORGOT PASSWORD?</Text>
          </TouchableOpacity>

          {/* ACCESS SYSTEM BUTTON */}
          <TouchableOpacity style={styles.accessBtn} onPress={() => navigation && navigation.navigate('Dashboard')}>
            <View style={[styles.accessBtnGradient, { paddingVertical: actionBtnVertical }]}> 
              <Text style={[styles.accessBtnText, { fontSize: primaryBtnTextSize, letterSpacing: isNarrow ? 1.2 : 2 }]}>→) ACCESS SYSTEM</Text>
            </View>
          </TouchableOpacity>

          {/* Divider */}
          <View style={styles.dividerRow}>
            <View style={styles.dividerLine} />
            <Text style={styles.dividerText}>OR QUICK ACCESS</Text>
            <View style={styles.dividerLine} />
          </View>

          {/* BIOMETRIC LOGIN */}
          <TouchableOpacity style={[styles.biometricBtn, { paddingVertical: actionBtnVertical }]}> 
            <Text style={[styles.biometricBtnText, { fontSize: secondaryBtnTextSize, letterSpacing: isNarrow ? 1.2 : 2 }]}>〜 BIOMETRIC LOGIN</Text>
          </TouchableOpacity>

          {/* VEIN SCAN ACCESS */}
          <TouchableOpacity style={[styles.veinBtn, { paddingVertical: actionBtnVertical }]}> 
            <Text style={[styles.veinBtnText, { fontSize: secondaryBtnTextSize, letterSpacing: isNarrow ? 1.2 : 2 }]}>✋ VEIN SCAN ACCESS</Text>
          </TouchableOpacity>
        </View>

        {/* Footer */}
        <View style={styles.footer}>
          <View style={styles.footerBadges}>
            <Text style={styles.footerBadge}>⬡ ESP32</Text>
            <Text style={styles.footerSep}>|</Text>
            <Text style={styles.footerBadge}>📶 MQTT</Text>
            <Text style={styles.footerSep}>|</Text>
            <Text style={styles.footerBadge}>🛡 AES-256</Text>
          </View>
          <Text style={styles.footerCopy}>© 2024 VeinGuard Systems. All Rights Reserved.</Text>
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
  scroll: { flexGrow: 1, alignItems: 'center' },
  contentWrap: { width: '100%' },

  // Corner decorations
  corner: { position: 'absolute', borderColor: COLORS.teal, zIndex: 10 },
  cornerTL: { top: 12, left: 12, borderTopWidth: 2, borderLeftWidth: 2 },
  cornerTR: { top: 12, right: 12, borderTopWidth: 2, borderRightWidth: 2 },
  cornerBL: { bottom: 12, left: 12, borderBottomWidth: 2, borderLeftWidth: 2 },
  cornerBR: { bottom: 12, right: 12, borderBottomWidth: 2, borderRightWidth: 2 },

  // Logo
  logoSection: { alignItems: 'center', marginBottom: 24 },
  logoIconCircle: {
    width: 64, height: 64, borderRadius: 32,
    borderWidth: 2, borderColor: COLORS.teal,
    backgroundColor: '#0a1e35',
    alignItems: 'center', justifyContent: 'center',
    marginBottom: 10,
    shadowColor: COLORS.teal, shadowOffset: { width: 0, height: 0 }, shadowOpacity: 0.8, shadowRadius: 10,
  },
  logoIconText: { color: COLORS.teal, fontSize: 28 },
  logoText: { fontSize: 30, fontWeight: '900', letterSpacing: 3, marginBottom: 4 },
  logoVein: { color: COLORS.white },
  logoGuard: { color: COLORS.teal },
  logoSubtitle: { color: COLORS.textDim, fontSize: 11, letterSpacing: 4, marginBottom: 14 },
  mqttBadge: {
    flexDirection: 'row', alignItems: 'center',
    borderWidth: 1, borderColor: COLORS.green,
    borderRadius: 20, paddingHorizontal: 12, paddingVertical: 4,
  },
  mqttDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: COLORS.green, marginRight: 6 },
  mqttText: { color: COLORS.green, fontSize: 11, fontWeight: '700', letterSpacing: 1 },

  // Form card
  formCard: {
    backgroundColor: COLORS.cardBg,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: COLORS.cardBorder,
    padding: 20,
    marginBottom: 24,
    position: 'relative',
  },
  accentBar: {
    position: 'absolute', left: 0, top: 16, bottom: 16,
    width: 3, backgroundColor: COLORS.teal, borderRadius: 2,
  },

  fieldGroup: { marginBottom: 12 },
  fieldLabel: { color: COLORS.teal, fontSize: 11, fontWeight: '700', letterSpacing: 1.5, marginBottom: 8 },
  fieldIcon: { color: COLORS.teal },
  input: {
    backgroundColor: COLORS.inputBg,
    borderWidth: 1, borderColor: COLORS.borderBlue,
    borderRadius: 8, paddingHorizontal: 14, paddingVertical: 12,
    color: COLORS.white, fontSize: 14,
  },
  passwordRow: { position: 'relative' },
  passwordInput: { paddingRight: 46 },
  eyeBtn: { position: 'absolute', right: 12, top: 12 },
  eyeIcon: { fontSize: 18 },

  domainRow: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    backgroundColor: COLORS.inputBg, borderWidth: 1, borderColor: COLORS.borderBlue,
    borderRadius: 8, padding: 12, marginBottom: 12,
  },
  domainRowCompact: {
    alignItems: 'flex-start',
    gap: 8,
  },
  domainLeft: { flexDirection: 'row', alignItems: 'center' },
  domainIcon: { fontSize: 16, marginRight: 8 },
  domainLabel: { color: COLORS.amber, fontSize: 11, fontWeight: '700', letterSpacing: 1, flexShrink: 1 },
  domainDropdown: {
    borderWidth: 1, borderColor: COLORS.amber,
    borderRadius: 6, paddingHorizontal: 12, paddingVertical: 6,
  },
  domainDropdownCompact: {
    alignSelf: 'flex-end',
  },
  domainDropdownText: { color: COLORS.amber, fontSize: 12, fontWeight: '600' },

  forgotRow: { alignItems: 'flex-end', marginBottom: 14 },
  forgotText: { color: COLORS.teal, fontSize: 12 },

  accessBtn: { borderRadius: 8, overflow: 'hidden', marginBottom: 14 },
  accessBtnGradient: {
    paddingVertical: 16, alignItems: 'center',
    backgroundColor: '#7000cc',
    borderWidth: 0,
  },
  accessBtnText: { color: COLORS.white, fontSize: 14, fontWeight: '800', letterSpacing: 2 },

  dividerRow: { flexDirection: 'row', alignItems: 'center', marginBottom: 12 },
  dividerLine: { flex: 1, height: 1, backgroundColor: COLORS.borderBlue },
  dividerText: { color: COLORS.textDim, fontSize: 10, marginHorizontal: 10, letterSpacing: 1 },

  biometricBtn: {
    borderWidth: 1.5, borderColor: COLORS.green,
    borderRadius: 8, paddingVertical: 14, alignItems: 'center', marginBottom: 10,
    backgroundColor: 'transparent',
  },
  biometricBtnText: { color: COLORS.green, fontSize: 13, fontWeight: '700', letterSpacing: 2 },

  veinBtn: {
    borderWidth: 1.5, borderColor: COLORS.magenta,
    borderRadius: 8, paddingVertical: 14, alignItems: 'center',
    backgroundColor: 'transparent',
  },
  veinBtnText: { color: COLORS.magenta, fontSize: 13, fontWeight: '700', letterSpacing: 2 },

  // Footer
  footer: { alignItems: 'center' },
  footerBadges: { flexDirection: 'row', alignItems: 'center', marginBottom: 6 },
  footerBadge: { color: COLORS.textDim, fontSize: 11 },
  footerSep: { color: COLORS.textDim, marginHorizontal: 8 },
  footerCopy: { color: COLORS.textDim, fontSize: 10 },
});
