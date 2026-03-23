import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  StatusBar,
  Platform,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { BlurView } from 'expo-blur';
import { COLORS, GRADIENTS } from '../theme';

function CheckItem({ label, checked, onToggle }) {
  return (
    <TouchableOpacity style={styles.checkRow} onPress={onToggle} activeOpacity={0.7}>
      <View style={[styles.checkbox, checked && { borderColor: COLORS.neonGreen, backgroundColor: 'rgba(57, 255, 20, 0.1)' }]}>
        {checked && <Ionicons name="checkmark" size={14} color={COLORS.neonGreen} />}
      </View>
      <Text style={[styles.checkText, checked && { color: COLORS.white }]}>{label}</Text>
    </TouchableOpacity>
  );
}

function StatusIndicator({ label, active }) {
  return (
    <View style={styles.statusItem}>
      <View style={[styles.statusDot, { backgroundColor: active ? COLORS.neonGreen : 'rgba(255,255,255,0.1)' }]} />
      <Text style={[styles.statusLabel, active && { color: COLORS.white }]}>{label.toUpperCase()}</Text>
    </View>
  );
}

import { useMqttStore } from '../store/mqttStore';
import { Alert } from 'react-native';

export default function EnrollUser({ navigation }) {
  const { t } = useTranslation();
  const [fullName, setFullName] = useState('John Doe');
  const [employeeId, setEmployeeId] = useState('EMP-001');
  const [email, setEmail] = useState('john.doe@corp.com');
  const [department, setDepartment] = useState('Engineering');
  const [notes, setNotes] = useState('');

  const [groupMain, setGroupMain] = useState(true);
  const [groupLab, setGroupLab] = useState(false);
  const [groupServer, setGroupServer] = useState(false);
  const [groupAdmin, setGroupAdmin] = useState(false);
  const [consent, setConsent] = useState(true);

  const isConnected = useMqttStore((state) => state.isConnected);
  const client = useMqttStore((state) => state.client);

  const handleCompleteEnrollment = async () => {
    if (!isConnected) {
      Alert.alert("System Offline", "Unable to reach the security gateway via MQTT.");
      return;
    }

    if (!consent) {
      Alert.alert("Consent Required", "You must agree to biometric data processing.");
      return;
    }

    try {
      // In a real scenario, we'd have real base64 images from the sensor.
      // For now, we send the intent to the backend.
      const payload = {
        user_id: employeeId,
        username: fullName,
        email: email,
        department: department,
        images: [] // To be populated by real sensor integration later
      };

      client.publish('veinguard/cmd/enroll', JSON.stringify(payload));
      
      Alert.alert(
        "Enrollment Initiated",
        "The security gateway has started the biometric provisioning cycle. Please follow the instructions on the hardware LCD.",
        [{ text: "OK", onPress: () => navigation.navigate('UserManagement') }]
      );
    } catch (err) {
      Alert.alert("Error", "Failed to transmit enrollment packet.");
    }
  };

  return (
    <View style={styles.screen}>
      <StatusBar barStyle="light-content" />
      <LinearGradient colors={GRADIENTS.primary} style={StyleSheet.absoluteFill} />

      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation?.goBack()} style={styles.backBtn}>
          <Ionicons name="chevron-back" size={24} color={COLORS.white} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>{t('enrollment.title')}</Text>
        <View style={styles.spacer} />
      </View>

      <ScrollView style={styles.scroll} showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>
        <View style={styles.welcomeSection}>
          <Text style={styles.subtitle}>{t('enrollment.subtitle')}</Text>
        </View>

        {/* Profile Section */}
        <BlurView intensity={15} tint="dark" style={styles.card}>
          <Text style={styles.cardTitle}>{t('enrollment.userProfile').toUpperCase()}</Text>
          
          <View style={styles.photoContainer}>
            <LinearGradient colors={['#1c3d5a', '#0d1b2e']} style={styles.photoCircle}>
              <Ionicons name="person-add" size={40} color={COLORS.neonCyan} />
            </LinearGradient>
            <TouchableOpacity style={styles.photoAction}>
              <Text style={styles.photoActionText}>{t('enrollment.capturePhoto').toUpperCase()}</Text>
            </TouchableOpacity>
          </View>

          <View style={styles.formRow}>
            <View style={styles.inputGroup}>
              <Text style={styles.inputLabel}>{t('enrollment.fullName').toUpperCase()}</Text>
              <TextInput style={styles.input} value={fullName} onChangeText={setFullName} placeholderTextColor={COLORS.textDim} />
            </View>
            <View style={styles.inputGroup}>
              <Text style={styles.inputLabel}>{t('enrollment.employeeId').toUpperCase()}</Text>
              <TextInput style={styles.input} value={employeeId} onChangeText={setEmployeeId} placeholderTextColor={COLORS.textDim} />
            </View>
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.inputLabel}>{t('enrollment.email').toUpperCase()}</Text>
            <TextInput style={styles.input} value={email} onChangeText={setEmail} keyboardType="email-address" />
          </View>

          <View style={styles.formRow}>
            <View style={styles.inputGroup}>
              <Text style={styles.inputLabel}>{t('enrollment.role').toUpperCase()}</Text>
              <View style={styles.pickerView}>
                <Text style={styles.pickerText}>OPERATOR</Text>
                <Ionicons name="chevron-down" size={14} color={COLORS.textDim} />
              </View>
            </View>
            <View style={styles.inputGroup}>
              <Text style={styles.inputLabel}>{t('enrollment.department').toUpperCase()}</Text>
              <TextInput style={styles.input} value={department} onChangeText={setDepartment} />
            </View>
          </View>

          <Text style={styles.inputLabel}>{t('enrollment.accessGroups').toUpperCase()}</Text>
          <View style={styles.checkGrid}>
            <CheckItem label="Main Entrance" checked={groupMain} onToggle={() => setGroupMain(!groupMain)} />
            <CheckItem label="Server Room" checked={groupServer} onToggle={() => setGroupServer(!groupServer)} />
            <CheckItem label="Lab Area" checked={groupLab} onToggle={() => setGroupLab(!groupLab)} />
            <CheckItem label="Admin Office" checked={groupAdmin} onToggle={() => setGroupAdmin(!groupAdmin)} />
          </View>
        </BlurView>

        {/* Biometric Capture Section */}
        <BlurView intensity={15} tint="dark" style={styles.card}>
          <View style={styles.cardHeader}>
            <Text style={[styles.cardTitle, { color: COLORS.neonPurple }]}>{t('enrollment.veinCapture').toUpperCase()}</Text>
            <Ionicons name="finger-print" size={20} color={COLORS.neonPurple} />
          </View>

          <View style={styles.scannerInterface}>
            <LinearGradient colors={['rgba(0, 242, 255, 0.05)', 'transparent']} style={styles.scannerGlow}>
              <Ionicons name="hand-left" size={60} color={COLORS.neonCyan} style={{ opacity: 0.5 }} />
              <Text style={styles.scannerHint}>ALIGN HAND WITHIN SENSOR RANGE</Text>
            </LinearGradient>
          </View>

          <TouchableOpacity style={styles.primaryAction}>
            <LinearGradient colors={['#1c3d5a', '#0d1b2e']} style={styles.primaryActionInner}>
              <Text style={styles.primaryActionText}>{t('enrollment.startCapture').toUpperCase()}</Text>
            </LinearGradient>
          </TouchableOpacity>

          <View style={styles.statusGrid}>
            <StatusIndicator label="Liveness" active={false} />
            <StatusIndicator label="Quality" active={false} />
            <StatusIndicator label="Pattern" active={false} />
          </View>
        </BlurView>

        {/* Consent Section */}
        <BlurView intensity={10} style={styles.consentCard}>
          <View style={styles.consentHeader}>
            <Ionicons name="shield-half" size={20} color={COLORS.neonAmber} />
            <Text style={styles.consentTitle}>{t('enrollment.biometricDataConsent').toUpperCase()}</Text>
          </View>
          <Text style={styles.consentBody}>{t('enrollment.consentBodyText')}</Text>
          <CheckItem label={t('enrollment.consentCheckLabel')} checked={consent} onToggle={() => setConsent(!consent)} />
        </BlurView>

        {/* Footer Actions */}
        <View style={styles.footer}>
          <TouchableOpacity style={styles.saveBtn} onPress={handleCompleteEnrollment}>
            <LinearGradient colors={GRADIENTS.neonCyan} style={styles.saveBtnInner}>
              <Text style={styles.saveBtnText}>{t('enrollment.completeEnrollment').toUpperCase()}</Text>
            </LinearGradient>
          </TouchableOpacity>
          <TouchableOpacity style={styles.cancelBtn} onPress={() => navigation?.goBack()}>
            <Text style={styles.cancelBtnText}>{t('common.cancel').toUpperCase()}</Text>
          </TouchableOpacity>
        </View>
        
        <View style={{ height: 40 }} />
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: COLORS.bg },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingTop: Platform.OS === 'ios' ? 50 : 20,
    paddingBottom: 20,
    paddingHorizontal: 20,
  },
  backBtn: { width: 40, height: 40, justifyContent: 'center' },
  headerTitle: { color: COLORS.white, fontSize: 18, fontWeight: '800', letterSpacing: 1 },
  spacer: { width: 40 },

  scroll: { flex: 1, paddingHorizontal: 20 },
  scrollContent: { paddingTop: 10 },
  welcomeSection: { marginBottom: 25 },
  subtitle: { color: COLORS.textSecondary, fontSize: 14 },

  card: {
    borderRadius: 24,
    padding: 20,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.05)',
    marginBottom: 20,
    overflow: 'hidden',
  },
  cardHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 },
  cardTitle: { color: COLORS.neonCyan, fontSize: 12, fontWeight: '900', letterSpacing: 2, marginBottom: 20 },
  
  photoContainer: { alignItems: 'center', marginBottom: 25 },
  photoCircle: { width: 90, height: 90, borderRadius: 45, justifyContent: 'center', alignItems: 'center', marginBottom: 15, borderWidth: 1, borderColor: 'rgba(0, 242, 255, 0.2)' },
  photoAction: { paddingHorizontal: 15, paddingVertical: 8, borderRadius: 10, backgroundColor: 'rgba(0, 242, 255, 0.1)', borderWidth: 1, borderColor: 'rgba(0, 242, 255, 0.2)' },
  photoActionText: { color: COLORS.neonCyan, fontSize: 10, fontWeight: '800', letterSpacing: 1 },

  formRow: { flexDirection: 'row', gap: 15, marginBottom: 15 },
  inputGroup: { flex: 1, marginBottom: 15 },
  inputLabel: { color: COLORS.textDim, fontSize: 9, fontWeight: '800', letterSpacing: 1, marginBottom: 8 },
  input: {
    backgroundColor: 'rgba(255, 255, 255, 0.03)',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.08)',
    color: COLORS.white,
    paddingHorizontal: 15,
    paddingVertical: 12,
    fontSize: 14,
    fontWeight: '600',
  },
  pickerView: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.03)',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.08)',
    paddingHorizontal: 15,
    paddingVertical: 12,
  },
  pickerText: { color: COLORS.white, fontSize: 14, fontWeight: '600' },

  checkGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 10 },
  checkRow: { flexDirection: 'row', alignItems: 'center', width: '47%', marginBottom: 5 },
  checkbox: { width: 18, height: 18, borderRadius: 5, borderWidth: 1, borderColor: 'rgba(255, 255, 255, 0.2)', marginRight: 10, justifyContent: 'center', alignItems: 'center' },
  checkText: { color: COLORS.textDim, fontSize: 12, fontWeight: '500' },

  scannerInterface: {
    height: 180,
    backgroundColor: 'rgba(0, 0, 0, 0.2)',
    borderRadius: 20,
    borderWidth: 1,
    borderColor: 'rgba(188, 19, 254, 0.1)',
    marginBottom: 20,
    overflow: 'hidden',
  },
  scannerGlow: { flex: 1, justifyContent: 'center', alignItems: 'center', gap: 15 },
  scannerHint: { color: COLORS.textDim, fontSize: 10, fontWeight: '700', letterSpacing: 1 },

  primaryAction: { borderRadius: 15, overflow: 'hidden', marginBottom: 20, borderWidth: 1, borderColor: 'rgba(188, 19, 254, 0.3)' },
  primaryActionInner: { paddingVertical: 15, alignItems: 'center' },
  primaryActionText: { color: COLORS.neonPurple, fontWeight: '900', letterSpacing: 2, fontSize: 13 },

  statusGrid: { flexDirection: 'row', justifyContent: 'space-between', paddingHorizontal: 10 },
  statusItem: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  statusDot: { width: 6, height: 6, borderRadius: 3 },
  statusLabel: { color: COLORS.textDim, fontSize: 9, fontWeight: '700' },

  consentCard: { padding: 20, borderRadius: 24, borderWidth: 1, borderColor: 'rgba(255, 216, 78, 0.1)', marginBottom: 30 },
  consentHeader: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 15 },
  consentTitle: { color: COLORS.neonAmber, fontSize: 11, fontWeight: '900', letterSpacing: 1 },
  consentBody: { color: COLORS.textDim, fontSize: 13, lineHeight: 20, marginBottom: 15 },

  footer: { gap: 12 },
  saveBtn: { borderRadius: 18, overflow: 'hidden' },
  saveBtnInner: { paddingVertical: 18, alignItems: 'center' },
  saveBtnText: { color: COLORS.white, fontWeight: '900', letterSpacing: 2, fontSize: 14 },
  cancelBtn: { paddingVertical: 15, alignItems: 'center' },
  cancelBtnText: { color: COLORS.textDim, fontWeight: '700', letterSpacing: 1, fontSize: 12 },
});