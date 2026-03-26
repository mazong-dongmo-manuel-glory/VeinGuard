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
  Alert,
  Image,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { BlurView } from 'expo-blur';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { COLORS, GRADIENTS } from '../theme';
import { getAppErrorMessage } from '../services/appErrors';
import { useMqttStore } from '../store/mqttStore';

const DEPARTMENT_KEYS = ['security', 'operations', 'research', 'it', 'administration'];

function CheckItem({ label, checked, onToggle }) {
  return (
    <TouchableOpacity style={styles.checkRow} onPress={onToggle} activeOpacity={0.7}>
      <View style={[styles.checkbox, checked && styles.checkboxActive]}>
        {checked && <Ionicons name="checkmark" size={14} color={COLORS.neonGreen} />}
      </View>
      <Text style={[styles.checkText, checked && styles.checkTextActive]}>{label}</Text>
    </TouchableOpacity>
  );
}

function StatusIndicator({ label, active }) {
  return (
    <View style={styles.statusItem}>
      <View style={[styles.statusDot, active ? styles.statusDotActive : null]} />
      <Text style={[styles.statusLabel, active && styles.statusLabelActive]}>{label.toUpperCase()}</Text>
    </View>
  );
}

export default function EnrollUser({ navigation, route }) {
  const { t } = useTranslation();
  const insets = useSafeAreaInsets();
  const params = route?.params || {};
  const mode = params.mode || 'create';
  const editingUser = params.user || null;

  const [fullName, setFullName] = useState(editingUser?.username || '');
  const [employeeId] = useState(editingUser?.id || '');
  const [email, setEmail] = useState(editingUser?.email || '');
  const [department, setDepartment] = useState(editingUser?.department || t('enrollment.departments.security'));
  const [notes, setNotes] = useState('');
  const [departmentMenuOpen, setDepartmentMenuOpen] = useState(false);

  const [groupMain, setGroupMain] = useState(true);
  const [groupLab, setGroupLab] = useState(false);
  const [groupServer, setGroupServer] = useState(false);
  const [groupAdmin, setGroupAdmin] = useState(false);
  const [consent, setConsent] = useState(true);
  const [submitting, setSubmitting] = useState(false);

  const isConnected = useMqttStore((state) => state.isConnected);
  const enrollUser = useMqttStore((state) => state.enrollUser);
  const updateUser = useMqttStore((state) => state.updateUser);
  const telemetry = useMqttStore((state) => state.telemetry);
  const statusPayload = useMqttStore((state) => state.statusPayload);

  const previewBase64 = telemetry?.camera?.processed_jpeg_base64 || telemetry?.camera?.preview_jpeg_base64;
  const previewUri = previewBase64 ? `data:image/jpeg;base64,${previewBase64}` : null;
  const currentRole = groupAdmin ? t('enrollment.adminRole') : t('enrollment.operatorRole');
  const accessGroups = [
    groupMain && 'main',
    groupLab && 'lab',
    groupServer && 'server',
    groupAdmin && 'admin',
  ].filter(Boolean);

  const departmentOptions = DEPARTMENT_KEYS.map((key) => ({
    key,
    label: t(`enrollment.departments.${key}`),
  }));

  const enrollmentProgress = statusPayload?.phase === 'ENROLLMENT'
    ? Number(statusPayload?.sample_index || 0)
    : 0;
  const enrollmentTarget = Number(statusPayload?.sample_count || 5);

  const handleCompleteEnrollment = async () => {
    if (!isConnected) {
      Alert.alert(t('enrollment.offlineTitle'), t('enrollment.offlineDesc'));
      return;
    }

    if (!consent) {
      Alert.alert(t('enrollment.consentRequiredTitle'), t('enrollment.consentRequiredDesc'));
      return;
    }

    if (!fullName.trim() || !email.trim()) {
      Alert.alert(t('common.error'), t('enrollment.requiredFieldsDesc'));
      return;
    }

    if (!String(email).includes('@')) {
      Alert.alert(t('common.error'), t('enrollment.invalidEmailDesc'));
      return;
    }

    setSubmitting(true);
    try {
      const payload = {
        ...(mode === 'edit' && employeeId ? { user_id: employeeId } : {}),
        username: fullName.trim(),
        password: 'Temp1234!',
        role: groupAdmin ? 'admin' : 'operator',
        email: email.trim(),
        department: department.trim(),
        access_groups: accessGroups,
        notes: notes.trim(),
        images: [],
      };

      const response = mode === 'edit'
        ? await updateUser(payload)
        : await enrollUser(payload);

      if (!response || response.status !== 'success') {
        throw new Error(response?.error || response?.reason || t('enrollment.enrollmentErrorDesc'));
      }

      const successTitle = mode === 'edit' ? t('enrollment.editSuccessTitle') : t('common.success');
      const successMessage = mode === 'edit'
        ? t('enrollment.editSuccessDesc')
        : t('enrollment.enrollmentSuccessDesc', { userId: response?.user_id || t('enrollment.autoAssigned') });

      Alert.alert(successTitle, successMessage, [{ text: t('common.ok'), onPress: () => navigation.goBack() }]);
    } catch (err) {
      Alert.alert(t('common.error'), getAppErrorMessage(t, err, 'enrollment.enrollmentErrorDesc'));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <View style={styles.screen}>
      <StatusBar barStyle="light-content" />
      <LinearGradient colors={GRADIENTS.primary} style={StyleSheet.absoluteFill} />

      <View style={[styles.header, { paddingTop: Math.max(insets.top + 8, 20) }]}>
        <TouchableOpacity onPress={() => navigation?.goBack()} style={styles.backBtn}>
          <Ionicons name="chevron-back" size={24} color={COLORS.white} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>{mode === 'edit' ? t('userManagement.editUser') : t('enrollment.title')}</Text>
        <View style={styles.spacer} />
      </View>

      <ScrollView style={styles.scroll} showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>
        <View style={styles.welcomeSection}>
          <Text style={styles.subtitle}>{t('enrollment.subtitle')}</Text>
        </View>

        <BlurView intensity={15} tint="dark" style={styles.card}>
          <Text style={styles.cardTitle}>{t('enrollment.userProfile').toUpperCase()}</Text>

          <View style={styles.photoContainer}>
            <LinearGradient colors={['#1c3d5a', '#0d1b2e']} style={styles.photoCircle}>
              <Ionicons name="person-add" size={40} color={COLORS.neonCyan} />
            </LinearGradient>
          </View>

          <View style={styles.formRow}>
            <View style={styles.inputGroup}>
              <Text style={styles.inputLabel}>{t('enrollment.fullName').toUpperCase()}</Text>
              <TextInput
                style={styles.input}
                value={fullName}
                onChangeText={setFullName}
                placeholderTextColor={COLORS.textDim}
              />
            </View>
            <View style={styles.inputGroup}>
              <Text style={styles.inputLabel}>{t('enrollment.autoIdLabel').toUpperCase()}</Text>
              <View style={styles.readOnlyInput}>
                <Text style={styles.readOnlyValue}>{employeeId || t('enrollment.autoAssigned')}</Text>
              </View>
              <Text style={styles.helperText}>{t('enrollment.autoIdDesc')}</Text>
            </View>
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.inputLabel}>{t('enrollment.email').toUpperCase()}</Text>
            <TextInput style={styles.input} value={email} onChangeText={setEmail} keyboardType="email-address" autoCapitalize="none" />
          </View>

          <View style={styles.formRow}>
            <View style={styles.inputGroup}>
              <Text style={styles.inputLabel}>{t('enrollment.role').toUpperCase()}</Text>
              <View style={styles.readOnlyInput}>
                <Text style={styles.readOnlyValue}>{currentRole}</Text>
              </View>
            </View>
            <View style={styles.inputGroup}>
              <Text style={styles.inputLabel}>{t('enrollment.department').toUpperCase()}</Text>
              <TouchableOpacity
                style={styles.pickerView}
                activeOpacity={0.85}
                onPress={() => setDepartmentMenuOpen((value) => !value)}
              >
                <Text style={styles.pickerText}>{department || t('enrollment.departmentSelect')}</Text>
                <Ionicons name={departmentMenuOpen ? 'chevron-up' : 'chevron-down'} size={14} color={COLORS.textDim} />
              </TouchableOpacity>
              {departmentMenuOpen ? (
                <View style={styles.dropdownMenu}>
                  {departmentOptions.map((option) => (
                    <TouchableOpacity
                      key={option.key}
                      style={styles.dropdownItem}
                      onPress={() => {
                        setDepartment(option.label);
                        setDepartmentMenuOpen(false);
                      }}
                    >
                      <Text style={[styles.dropdownItemText, department === option.label && styles.dropdownItemTextActive]}>
                        {option.label}
                      </Text>
                    </TouchableOpacity>
                  ))}
                </View>
              ) : null}
            </View>
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.inputLabel}>{t('enrollment.notes').toUpperCase()}</Text>
            <TextInput
              style={[styles.input, styles.textArea]}
              value={notes}
              onChangeText={setNotes}
              multiline
              numberOfLines={3}
            />
          </View>

          <Text style={styles.inputLabel}>{t('enrollment.accessGroups').toUpperCase()}</Text>
          <View style={styles.checkGrid}>
            <CheckItem label={t('enrollment.mainEntrance')} checked={groupMain} onToggle={() => setGroupMain(!groupMain)} />
            <CheckItem label={t('enrollment.serverRoom')} checked={groupServer} onToggle={() => setGroupServer(!groupServer)} />
            <CheckItem label={t('enrollment.labArea')} checked={groupLab} onToggle={() => setGroupLab(!groupLab)} />
            <CheckItem label={t('enrollment.adminOffice')} checked={groupAdmin} onToggle={() => setGroupAdmin(!groupAdmin)} />
          </View>
        </BlurView>

        <BlurView intensity={15} tint="dark" style={styles.card}>
          <View style={styles.cardHeader}>
            <Text style={[styles.cardTitle, styles.captureTitle]}>{t('enrollment.veinCapture').toUpperCase()}</Text>
            <Ionicons name="scan-outline" size={20} color={COLORS.neonPurple} />
          </View>

          <View style={styles.scannerInterface}>
            {previewUri ? (
              <Image source={{ uri: previewUri }} style={styles.previewImage} resizeMode="cover" />
            ) : (
              <LinearGradient colors={['rgba(0, 242, 255, 0.05)', 'transparent']} style={styles.scannerGlow}>
                <Ionicons name="hand-left" size={60} color={COLORS.neonCyan} style={styles.handIcon} />
                <Text style={styles.scannerHint}>{t('enrollment.cameraPreviewUnavailable')}</Text>
              </LinearGradient>
            )}
            <View style={styles.scannerOverlay}>
              <View style={styles.cornerTL} />
              <View style={styles.cornerTR} />
              <View style={styles.cornerBL} />
              <View style={styles.cornerBR} />
            </View>
          </View>

          <Text style={styles.livePreviewLabel}>{t('enrollment.livePreview')}</Text>
          <Text style={styles.captureDesc}>{t('enrollment.multiAngleDesc')}</Text>
          <Text style={styles.sampleCountText}>
            {t('enrollment.sampleCountLabel')}: {enrollmentProgress > 0 ? `${enrollmentProgress}/${enrollmentTarget}` : String(enrollmentTarget)}
          </Text>

          <View style={styles.statusGrid}>
            <StatusIndicator label={t('enrollment.livePreview')} active={Boolean(previewUri)} />
            <StatusIndicator label={t('enrollment.qualityStatus')} active={Boolean(telemetry?.camera?.available)} />
            <StatusIndicator label={t('enrollment.pattern')} active={enrollmentProgress > 0} />
          </View>
        </BlurView>

        <BlurView intensity={10} style={styles.consentCard}>
          <View style={styles.consentHeader}>
            <Ionicons name="shield-half" size={20} color={COLORS.neonAmber} />
            <Text style={styles.consentTitle}>{t('enrollment.biometricDataConsent').toUpperCase()}</Text>
          </View>
          <Text style={styles.consentBody}>{t('enrollment.consentBodyText')}</Text>
          <CheckItem label={t('enrollment.consentCheckLabel')} checked={consent} onToggle={() => setConsent(!consent)} />
        </BlurView>

        <View style={styles.footer}>
          <TouchableOpacity style={[styles.saveBtn, submitting && styles.saveBtnDisabled]} onPress={handleCompleteEnrollment} disabled={submitting}>
            <LinearGradient colors={GRADIENTS.neonCyan} style={styles.saveBtnInner}>
              <Text style={styles.saveBtnText}>
                {submitting
                  ? t('enrollment.startCapture').toUpperCase()
                  : mode === 'edit'
                    ? t('userManagement.editUser')
                    : t('enrollment.completeEnrollment').toUpperCase()}
              </Text>
            </LinearGradient>
          </TouchableOpacity>
          <TouchableOpacity style={styles.cancelBtn} onPress={() => navigation?.goBack()}>
            <Text style={styles.cancelBtnText}>{t('common.cancel').toUpperCase()}</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.bottomSpacer} />
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
    paddingBottom: 20,
    paddingHorizontal: 20,
  },
  backBtn: { width: 40, height: 40, justifyContent: 'center' },
  headerTitle: { color: COLORS.white, fontSize: 18, fontWeight: '800', letterSpacing: 1 },
  spacer: { width: 40 },
  scroll: { flex: 1, paddingHorizontal: 20 },
  scrollContent: { paddingTop: 10 },
  welcomeSection: { marginBottom: 25 },
  subtitle: { color: COLORS.textSecondary, fontSize: 14, lineHeight: 20 },
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
  captureTitle: { marginBottom: 0 },
  photoContainer: { alignItems: 'center', marginBottom: 20 },
  photoCircle: {
    width: 90,
    height: 90,
    borderRadius: 45,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(0, 242, 255, 0.2)',
  },
  formRow: { flexDirection: 'row', gap: 15, marginBottom: 15, flexWrap: 'wrap' },
  inputGroup: { flex: 1, marginBottom: 15, minWidth: 140 },
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
  readOnlyInput: {
    backgroundColor: 'rgba(255,255,255,0.03)',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.08)',
    minHeight: 48,
    justifyContent: 'center',
    paddingHorizontal: 15,
  },
  readOnlyValue: { color: COLORS.white, fontSize: 14, fontWeight: '700' },
  helperText: { color: COLORS.textSecondary, fontSize: 11, lineHeight: 16, marginTop: 8 },
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
  pickerText: { color: COLORS.white, fontSize: 14, fontWeight: '600', flexShrink: 1 },
  dropdownMenu: {
    marginTop: 8,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.08)',
    backgroundColor: 'rgba(9, 18, 31, 0.96)',
    overflow: 'hidden',
  },
  dropdownItem: {
    paddingHorizontal: 14,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255,255,255,0.04)',
  },
  dropdownItemText: { color: COLORS.textSecondary, fontSize: 13, fontWeight: '700' },
  dropdownItemTextActive: { color: COLORS.neonCyan },
  textArea: { minHeight: 86, textAlignVertical: 'top' },
  checkGrid: { gap: 10 },
  checkRow: { flexDirection: 'row', alignItems: 'flex-start', width: '100%', marginBottom: 5 },
  checkbox: {
    width: 18,
    height: 18,
    borderRadius: 5,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
    marginRight: 10,
    justifyContent: 'center',
    alignItems: 'center',
  },
  checkboxActive: { borderColor: COLORS.neonGreen, backgroundColor: 'rgba(57, 255, 20, 0.1)' },
  checkText: { color: COLORS.textDim, fontSize: 12, fontWeight: '500', flex: 1, lineHeight: 18 },
  checkTextActive: { color: COLORS.white },
  scannerInterface: {
    height: 220,
    backgroundColor: 'rgba(0, 0, 0, 0.22)',
    borderRadius: 20,
    borderWidth: 1,
    borderColor: 'rgba(188, 19, 254, 0.1)',
    marginBottom: 16,
    overflow: 'hidden',
    position: 'relative',
  },
  previewImage: { width: '100%', height: '100%' },
  scannerGlow: { flex: 1, justifyContent: 'center', alignItems: 'center', gap: 15, paddingHorizontal: 18 },
  handIcon: { opacity: 0.55 },
  scannerHint: {
    color: COLORS.textDim,
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 0.8,
    textAlign: 'center',
    lineHeight: 16,
  },
  scannerOverlay: { ...StyleSheet.absoluteFillObject },
  cornerTL: { position: 'absolute', top: 15, left: 15, width: 26, height: 26, borderTopWidth: 2, borderLeftWidth: 2, borderColor: COLORS.neonCyan },
  cornerTR: { position: 'absolute', top: 15, right: 15, width: 26, height: 26, borderTopWidth: 2, borderRightWidth: 2, borderColor: COLORS.neonCyan },
  cornerBL: { position: 'absolute', bottom: 15, left: 15, width: 26, height: 26, borderBottomWidth: 2, borderLeftWidth: 2, borderColor: COLORS.neonCyan },
  cornerBR: { position: 'absolute', bottom: 15, right: 15, width: 26, height: 26, borderBottomWidth: 2, borderRightWidth: 2, borderColor: COLORS.neonCyan },
  livePreviewLabel: { color: COLORS.neonCyan, fontSize: 12, fontWeight: '800', marginBottom: 8 },
  captureDesc: { color: COLORS.textSecondary, fontSize: 12, lineHeight: 18, marginBottom: 8 },
  sampleCountText: { color: COLORS.white, fontSize: 12, fontWeight: '800', marginBottom: 16 },
  statusGrid: { flexDirection: 'row', justifyContent: 'space-between', paddingHorizontal: 6, flexWrap: 'wrap', gap: 10 },
  statusItem: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  statusDot: { width: 6, height: 6, borderRadius: 3, backgroundColor: 'rgba(255,255,255,0.12)' },
  statusDotActive: { backgroundColor: COLORS.neonGreen },
  statusLabel: { color: COLORS.textDim, fontSize: 10, fontWeight: '800', letterSpacing: 1 },
  statusLabelActive: { color: COLORS.white },
  consentCard: {
    borderRadius: 20,
    padding: 18,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.05)',
    marginBottom: 20,
  },
  consentHeader: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 12 },
  consentTitle: { color: COLORS.neonAmber, fontSize: 11, fontWeight: '900', letterSpacing: 1.5, flex: 1 },
  consentBody: { color: COLORS.textSecondary, fontSize: 12, lineHeight: 18, marginBottom: 14 },
  footer: { gap: 12 },
  saveBtn: { borderRadius: 16, overflow: 'hidden' },
  saveBtnDisabled: { opacity: 0.7 },
  saveBtnInner: { paddingVertical: 17, alignItems: 'center', justifyContent: 'center' },
  saveBtnText: { color: COLORS.white, fontWeight: '900', letterSpacing: 1.5, fontSize: 13, textAlign: 'center', paddingHorizontal: 12 },
  cancelBtn: { paddingVertical: 14, alignItems: 'center' },
  cancelBtnText: { color: COLORS.textSecondary, fontWeight: '800', letterSpacing: 1.5 },
  bottomSpacer: { height: 40 },
});
