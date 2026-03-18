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
} from 'react-native';

const COLORS = {
  bg: '#080e1a',
  cardBg: '#0d1b2e',
  cardBorder: '#1a3a5c',
  green: '#00ff88',
  teal: '#00e5ff',
  amber: '#e6a020',
  red: '#ff3d5a',
  magenta: '#d400ff',
  text: '#b8cfe0',
  textDim: '#4a6a8a',
  white: '#ffffff',
  headerBg: '#0a1525',
  inputBg: '#091525',
};

function CheckItem({ label, checked, onToggle }) {
  return (
    <TouchableOpacity style={styles.checkRow} onPress={onToggle} activeOpacity={0.85}>
      <View style={[styles.checkbox, checked && styles.checkboxChecked]}>
        {checked ? <Text style={styles.checkboxTick}>✓</Text> : null}
      </View>
      <Text style={styles.checkText}>{label}</Text>
    </TouchableOpacity>
  );
}

function InfoStat({ title, value }) {
  return (
    <View style={styles.statusCol}>
      <Text style={styles.statusTitle}>{title}</Text>
      <Text style={styles.statusValue}>{value}</Text>
    </View>
  );
}

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

  return (
    <View style={styles.screen}>
      <StatusBar barStyle="light-content" backgroundColor={COLORS.headerBg} />

      <View style={styles.header}>
        <View style={styles.headerLeft}>
          <Text style={styles.logoVein}>VEIN</Text>
          <Text style={styles.logoGuard}>GUARD</Text>
          <View style={styles.mqttBadge}>
            <View style={styles.mqttDot} />
            <Text style={styles.mqttText}>MQTT ONLINE</Text>
          </View>
        </View>
        <View style={styles.headerRight}>
          <Text style={styles.headerTime}>12:19:13   UTC+4</Text>
          <View style={styles.avatarCircle}><Text style={styles.avatarEmoji}>👤</Text></View>
          <Text style={styles.adminText}>Admin</Text>
        </View>
      </View>

      <View style={styles.dropdown}>
        <Text style={styles.dropdownText}>User Management (Roles/List)</Text>
        <Text style={styles.dropdownArrow}>▼</Text>
      </View>

      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
        keyboardShouldPersistTaps="handled"
      >
        <View style={styles.titleRow}>
          <TouchableOpacity
            style={styles.backBtn}
            onPress={() => navigation?.goBack()}
            activeOpacity={0.85}
          >
            <Text style={styles.backIcon}>←</Text>
          </TouchableOpacity>
          <View style={styles.titleTextWrap}>
            <Text style={styles.pageTitle}>ADD / ENROLL USER</Text>
            <Text style={styles.pageSubtitle}>Complete user profile and biometric enrollment</Text>
          </View>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>👥 USER PROFILE</Text>

          <View style={styles.photoWrap}>
            <View style={styles.photoCircle}><Text style={styles.photoEmoji}>👩</Text></View>
            <TouchableOpacity style={styles.captureBtn}>
              <Text style={styles.captureBtnText}>▣ CAPTURE PHOTO</Text>
            </TouchableOpacity>
          </View>

          <View style={styles.row2}>
            <View style={styles.fieldHalf}>
              <Text style={styles.fieldLabel}>FULL NAME</Text>
              <TextInput style={styles.input} value={fullName} onChangeText={setFullName} />
            </View>
            <View style={styles.fieldHalf}>
              <Text style={styles.fieldLabel}>EMPLOYEE ID</Text>
              <TextInput style={styles.input} value={employeeId} onChangeText={setEmployeeId} />
            </View>
          </View>

          <Text style={styles.fieldLabel}>EMAIL</Text>
          <TextInput style={styles.input} value={email} onChangeText={setEmail} />

          <View style={styles.row2}>
            <View style={styles.fieldHalf}>
              <Text style={styles.fieldLabel}>ROLE</Text>
              <View style={styles.selectInput}>
                <Text style={styles.selectText}>Select Role</Text>
                <Text style={styles.selectArrow}>▼</Text>
              </View>
            </View>
            <View style={styles.fieldHalf}>
              <Text style={styles.fieldLabel}>DEPARTMENT</Text>
              <TextInput style={styles.input} value={department} onChangeText={setDepartment} />
            </View>
          </View>

          <Text style={styles.fieldLabel}>ACCESS GROUPS</Text>
          <View style={styles.groupGrid}>
            <CheckItem label="Main Entrance" checked={groupMain} onToggle={() => setGroupMain(!groupMain)} />
            <CheckItem label="Server Room" checked={groupServer} onToggle={() => setGroupServer(!groupServer)} />
            <CheckItem label="Lab Area" checked={groupLab} onToggle={() => setGroupLab(!groupLab)} />
            <CheckItem label="Admin Office" checked={groupAdmin} onToggle={() => setGroupAdmin(!groupAdmin)} />
          </View>

          <Text style={styles.fieldLabel}>NOTES</Text>
          <TextInput
            style={[styles.input, styles.notesInput]}
            placeholder="Additional notes..."
            placeholderTextColor={COLORS.textDim}
            value={notes}
            onChangeText={setNotes}
            multiline
          />
        </View>

        <View style={styles.card}>
          <Text style={[styles.cardTitle, { color: COLORS.magenta }]}>✋ VEIN CAPTURE</Text>

          <View style={styles.scannerBox}>
            <Text style={styles.scannerHand}>✋</Text>
            <Text style={styles.scannerHint}>Place hand on scanner</Text>
          </View>

          <TouchableOpacity style={styles.startBtn}>
            <Text style={styles.startBtnText}>▶ START CAPTURE</Text>
          </TouchableOpacity>

          <View style={styles.captureMeta}>
            <InfoStat title="Liveness Detection" value="● PENDING" />
            <InfoStat title="Image Quality" value="● PENDING" />
            <InfoStat title="Vein Pattern" value="● PENDING" />
          </View>
        </View>

        <View style={styles.card}>
          <Text style={[styles.cardTitle, { color: COLORS.amber }]}>📶 ENROLLMENT STATUS</Text>
          <View style={styles.statusRow}>
            <View style={styles.statusBox}>
              <Text style={styles.statusDot}>○</Text>
              <Text style={styles.statusBoxTitle}>MQTT PUBLISH</Text>
              <Text style={styles.statusBoxSub}>Waiting for enrollment data...</Text>
            </View>
            <View style={styles.statusBox}>
              <Text style={styles.statusDot}>○</Text>
              <Text style={styles.statusBoxTitle}>DEVICE ACK</Text>
              <Text style={styles.statusBoxSub}>Awaiting device response...</Text>
            </View>
            <View style={styles.statusBox}>
              <Text style={styles.statusDot}>○</Text>
              <Text style={styles.statusBoxTitle}>SECURE STORAGE</Text>
              <Text style={styles.statusBoxSub}>Pending data encryption...</Text>
            </View>
          </View>
        </View>

        <View style={[styles.card, styles.consentCard]}>
          <Text style={[styles.cardTitle, styles.consentTitle]}>■ BIOMETRIC DATA CONSENT</Text>
          <CheckItem
            label="I consent to the collection, processing, and secure storage of my biometric data for access control purposes."
            checked={consent}
            onToggle={() => setConsent(!consent)}
          />
          <Text style={styles.consentText}>
            This data will be encrypted and stored securely, used only for authentication, and may be deleted upon request in accordance with privacy regulations.
          </Text>
        </View>

        <View style={styles.footerActions}>
          <TouchableOpacity style={styles.completeBtn}>
            <Text style={styles.completeBtnText}>✓ COMPLETE ENROLLMENT</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.cancelBtn} onPress={() => navigation?.goBack()}>
            <Text style={styles.cancelBtnText}>✕ CANCEL</Text>
          </TouchableOpacity>
        </View>

        <View style={{ height: 28 }} />
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
    backgroundColor: COLORS.headerBg,
    paddingHorizontal: 14,
    paddingTop: 44,
    paddingBottom: 10,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.cardBorder,
  },
  headerLeft: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  logoVein: { color: COLORS.white, fontWeight: '900', fontSize: 17, letterSpacing: 1 },
  logoGuard: { color: COLORS.teal, fontWeight: '900', fontSize: 17, letterSpacing: 1, marginRight: 10 },
  mqttBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: COLORS.green,
    borderRadius: 20,
    paddingHorizontal: 8,
    paddingVertical: 3,
  },
  mqttDot: { width: 7, height: 7, borderRadius: 3.5, backgroundColor: COLORS.green, marginRight: 4 },
  mqttText: { color: COLORS.green, fontSize: 9, fontWeight: '700', letterSpacing: 0.5 },
  headerRight: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  headerTime: { color: COLORS.textDim, fontSize: 10 },
  avatarCircle: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: COLORS.cardBorder,
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatarEmoji: { fontSize: 13 },
  adminText: { color: COLORS.text, fontSize: 11 },
  dropdown: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginHorizontal: 14,
    marginTop: 8,
    marginBottom: 8,
    padding: 10,
    backgroundColor: COLORS.cardBg,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: COLORS.cardBorder,
  },
  dropdownText: { color: COLORS.teal, fontSize: 12 },
  dropdownArrow: { color: COLORS.textDim, fontSize: 10 },
  scroll: { flex: 1, paddingHorizontal: 14 },
  scrollContent: { paddingBottom: 96 },
  titleRow: { flexDirection: 'row', alignItems: 'center', marginBottom: 12, marginTop: 4 },
  backBtn: {
    width: 26,
    height: 26,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: COLORS.teal,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 10,
    backgroundColor: '#072136',
  },
  backIcon: { color: COLORS.teal, fontSize: 13, fontWeight: '800' },
  titleTextWrap: { flex: 1 },
  pageTitle: { color: COLORS.white, fontSize: 31, lineHeight: 32, fontWeight: '900', letterSpacing: 1.4 },
  pageSubtitle: { color: COLORS.textDim, fontSize: 10, marginTop: 2 },
  card: {
    backgroundColor: COLORS.cardBg,
    borderRadius: 9,
    borderWidth: 1,
    borderColor: COLORS.cardBorder,
    padding: 12,
    marginBottom: 12,
  },
  cardTitle: { color: COLORS.teal, fontSize: 22, fontWeight: '800', letterSpacing: 1, marginBottom: 10 },
  photoWrap: { alignItems: 'center', marginBottom: 10 },
  photoCircle: {
    width: 86,
    height: 86,
    borderRadius: 43,
    borderWidth: 1.5,
    borderColor: COLORS.teal,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 8,
    backgroundColor: '#102238',
  },
  photoEmoji: { fontSize: 40 },
  captureBtn: {
    borderWidth: 1,
    borderColor: COLORS.teal,
    borderRadius: 4,
    paddingHorizontal: 10,
    paddingVertical: 5,
    backgroundColor: '#0b2f47',
  },
  captureBtnText: { color: COLORS.teal, fontSize: 9, fontWeight: '800' },
  row2: { flexDirection: 'row', gap: 8, marginBottom: 8 },
  fieldHalf: { flex: 1 },
  fieldLabel: { color: COLORS.textDim, fontSize: 10, marginBottom: 4, marginTop: 2 },
  input: {
    backgroundColor: COLORS.inputBg,
    borderColor: COLORS.cardBorder,
    borderWidth: 1,
    borderRadius: 4,
    color: COLORS.white,
    paddingHorizontal: 10,
    paddingVertical: 8,
    fontSize: 11,
    marginBottom: 8,
  },
  selectInput: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: COLORS.inputBg,
    borderColor: COLORS.cardBorder,
    borderWidth: 1,
    borderRadius: 4,
    paddingHorizontal: 10,
    paddingVertical: 8,
  },
  selectText: { color: COLORS.white, fontSize: 11 },
  selectArrow: { color: COLORS.textDim, fontSize: 9 },
  groupGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  checkRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    width: '48%',
    marginBottom: 6,
  },
  checkbox: {
    width: 12,
    height: 12,
    borderWidth: 1,
    borderColor: COLORS.textDim,
    borderRadius: 2,
    marginTop: 2,
    marginRight: 6,
    alignItems: 'center',
    justifyContent: 'center',
  },
  checkboxChecked: {
    backgroundColor: '#11384f',
    borderColor: COLORS.teal,
  },
  checkboxTick: { color: COLORS.teal, fontSize: 9, fontWeight: '800', lineHeight: 10 },
  checkText: { color: COLORS.text, fontSize: 10, flex: 1, lineHeight: 14 },
  notesInput: { minHeight: 48, textAlignVertical: 'top' },
  scannerBox: {
    borderWidth: 1,
    borderColor: '#2f5e85',
    borderRadius: 6,
    minHeight: 126,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#1d254f',
    marginBottom: 8,
  },
  scannerHand: { color: COLORS.teal, fontSize: 24, marginBottom: 4 },
  scannerHint: { color: COLORS.textDim, fontSize: 10 },
  startBtn: {
    borderWidth: 1,
    borderColor: COLORS.green,
    borderRadius: 7,
    paddingVertical: 10,
    alignItems: 'center',
    backgroundColor: '#103a24',
    shadowColor: COLORS.green,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.45,
    shadowRadius: 8,
    elevation: 2,
    marginBottom: 10,
  },
  startBtnText: { color: COLORS.green, fontSize: 10, fontWeight: '800', letterSpacing: 1 },
  captureMeta: {
    borderTopWidth: 1,
    borderTopColor: '#163a58',
    paddingTop: 8,
    gap: 6,
  },
  statusCol: { flexDirection: 'row', justifyContent: 'space-between' },
  statusTitle: { color: COLORS.textDim, fontSize: 10 },
  statusValue: { color: COLORS.textDim, fontSize: 10 },
  statusRow: { flexDirection: 'row', gap: 8 },
  statusBox: {
    flex: 1,
    backgroundColor: '#0a1426',
    borderRadius: 5,
    borderWidth: 1,
    borderColor: '#122b45',
    padding: 8,
    minHeight: 68,
  },
  statusDot: { color: COLORS.textDim, fontSize: 10, marginBottom: 4 },
  statusBoxTitle: { color: COLORS.text, fontSize: 9, fontWeight: '700', marginBottom: 3 },
  statusBoxSub: { color: COLORS.textDim, fontSize: 8, lineHeight: 12 },
  consentCard: {
    borderColor: '#5d4b20',
    backgroundColor: '#161716',
  },
  consentTitle: { color: '#ffd84e', fontSize: 11, marginBottom: 8 },
  consentText: { color: COLORS.textDim, fontSize: 9, lineHeight: 14, marginTop: 4 },
  footerActions: {
    flexDirection: 'row',
    gap: 8,
  },
  completeBtn: {
    flex: 1,
    borderWidth: 1,
    borderColor: COLORS.green,
    backgroundColor: '#103a24',
    borderRadius: 7,
    paddingVertical: 11,
    alignItems: 'center',
  },
  completeBtnText: { color: COLORS.green, fontSize: 11, fontWeight: '800', letterSpacing: 0.8 },
  cancelBtn: {
    width: 102,
    borderWidth: 1,
    borderColor: COLORS.red,
    backgroundColor: '#3a131a',
    borderRadius: 7,
    paddingVertical: 11,
    alignItems: 'center',
  },
  cancelBtnText: { color: '#ff7878', fontSize: 11, fontWeight: '800', letterSpacing: 0.8 },
});