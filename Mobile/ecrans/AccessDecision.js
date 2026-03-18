import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Image,
  StatusBar,
  Dimensions,
} from 'react-native';
import { useTranslation } from 'react-i18next';

const { width } = Dimensions.get('window');

// ─── Palette ────────────────────────────────────────────────────────────────
const COLORS = {
  bg: '#0a1628',
  cardBg: '#0f2035',
  cardBorder: '#1a3a5c',
  green: '#00ff88',
  greenDim: '#00c96a',
  greenDark: '#003d22',
  teal: '#00d4d4',
  amber: '#e6a020',
  amberDark: '#3d2800',
  red: '#e63946',
  redDark: '#3d0010',
  text: '#cdd6e0',
  textDim: '#6b8099',
  white: '#ffffff',
  headerBg: '#0d1f35',
  rowBg: '#0c1e31',
  sectionHeader: '#112233',
};

// ─── Helper Components ───────────────────────────────────────────────────────

const Header = () => (
  <View style={styles.header}>
    <View style={styles.headerLeft}>
      <Text style={styles.logoIcon}>{'〜'}</Text>
      <Text style={styles.logoText}>
        <Text style={styles.logoVein}>VEIN</Text>
        <Text style={styles.logoGuard}>GUARD</Text>
      </Text>
    </View>
    <View style={styles.avatarCircle}>
      <Text style={styles.avatarText}>👤</Text>
    </View>
  </View>
);

const DropdownBar = ({ t }) => (
  <View style={styles.dropdown}>
    <Text style={styles.dropdownText}>{t('accessDecision.dropdownLabel')}</Text>
    <Text style={styles.dropdownArrow}>▼</Text>
  </View>
);

const AccessGrantedBadge = ({ t }) => (
  <View style={styles.badgeContainer}>
    {/* Glowing circle with checkmark */}
    <View style={styles.checkCircleOuter}>
      <View style={styles.checkCircleInner}>
        <Text style={styles.checkMark}>✓</Text>
      </View>
    </View>
    <Text style={styles.accessGrantedText}>{t('accessDecision.title')}</Text>
    <Text style={styles.accessSubText}>{t('login.subtitle')}</Text>
  </View>
);

const UserCard = ({ t }) => (
  <View style={styles.card}>
    {/* User info header */}
    <View style={styles.userRow}>
      <View style={styles.userAvatar}>
        <Text style={styles.userAvatarText}>🧑‍💼</Text>
      </View>
      <View style={styles.userInfo}>
        <Text style={styles.userName}>John Mitchell</Text>
        <Text style={styles.userRole}>{t('accessDecision.userInfo')}</Text>
      </View>
    </View>
    {/* Divider */}
    <View style={styles.divider} />
    {/* Details */}
    <View style={styles.detailRow}>
      <Text style={styles.detailLabel}>{t('accessDecision.userId')}</Text>
      <Text style={styles.detailValue}>USR-2847</Text>
    </View>
    <View style={styles.detailRow}>
      <Text style={styles.detailLabel}>{t('common.admin')}</Text>
      <Text style={styles.detailValue}>Engineering</Text>
    </View>
    <View style={styles.detailRow}>
      <Text style={styles.detailLabel}>{t('accessDecision.accessLevel')}</Text>
      <Text style={[styles.detailValue, styles.greenText]}>Level 3</Text>
    </View>
  </View>
);

const ConfidenceScore = ({ t }) => (
  <View style={styles.card}>
    <View style={styles.confidenceHeader}>
      <Text style={styles.detailLabel}>{t('accessDecision.confidenceScore')}</Text>
      <Text style={styles.confidenceValue}>98.7%</Text>
    </View>
    <View style={styles.progressBarBg}>
      <View style={[styles.progressBarFill, { width: '98.7%' }]} />
    </View>
  </View>
);

const DeviceInfo = ({ t }) => (
  <View style={styles.card}>
    <View style={styles.deviceGrid}>
      <View style={styles.deviceCell}>
        <Text style={styles.deviceLabel}>{t('accessDecision.deviceId')}</Text>
        <Text style={styles.deviceValue}>ESP32-01</Text>
      </View>
      <View style={styles.deviceCell}>
        <Text style={styles.deviceLabel}>{t('accessDecision.doorId')}</Text>
        <Text style={styles.deviceValue}>DOOR-A12</Text>
      </View>
      <View style={styles.deviceCell}>
        <Text style={styles.deviceLabel}>{t('accessDecision.timestamp')}</Text>
        <Text style={styles.deviceValue}>10:28:00</Text>
      </View>
      <View style={styles.deviceCell}>
        <Text style={styles.deviceLabel}>{t('accessDecision.eventId')}</Text>
        <Text style={styles.deviceValue}>EVT-8743</Text>
      </View>
    </View>
  </View>
);

const ActionButtons = ({ t }) => (
  <View style={styles.actionsContainer}>
    <TouchableOpacity style={styles.btnViewEvent}>
      <Text style={styles.btnViewEventText}>{t('accessDecision.viewEventDetails')}</Text>
    </TouchableOpacity>
    <TouchableOpacity style={styles.btnRetry}>
      <Text style={styles.btnRetryText}>{t('accessDecision.retryScan')}</Text>
    </TouchableOpacity>
    <TouchableOpacity style={styles.btnReport}>
      <Text style={styles.btnReportText}>{t('accessDecision.reportIssue')}</Text>
    </TouchableOpacity>
  </View>
);

const AudioFeedback = ({ t }) => {
  const [on, setOn] = useState(true);
  return (
    <View style={styles.sectionCard}>
      <View style={styles.sectionTitleRow}>
        <Text style={styles.sectionIcon}>🔊</Text>
        <Text style={styles.sectionTitle}>{t('accessDecision.audioFeedback')}</Text>
      </View>
      <View style={styles.sectionDivider} />
      <View style={styles.sectionRow}>
        <Text style={styles.sectionRowLabel}>{t('accessDecision.audibleConfirmation')}</Text>
        <TouchableOpacity
          style={[styles.toggleBtn, on ? styles.toggleOn : styles.toggleOff]}
          onPress={() => setOn(!on)}
        >
          <Text style={styles.toggleText}>{on ? t('accessDecision.audioOn') : t('accessDecision.audioOff')}</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const DoorStatus = ({ t }) => {
  const [locked, setLocked] = useState(false);
  return (
    <View style={styles.sectionCard}>
      <View style={styles.sectionTitleRow}>
        <Text style={styles.sectionIcon}>🚪</Text>
        <Text style={styles.sectionTitle}>{t('accessDecision.doorStatus')}</Text>
      </View>
      <View style={styles.sectionDivider} />
      <View style={styles.sectionRow}>
        <Text style={styles.sectionRowLabel}>{t('accessDecision.doorLock')}</Text>
        <TouchableOpacity
          style={[styles.toggleBtn, !locked ? styles.toggleOn : styles.toggleOff]}
          onPress={() => setLocked(!locked)}
        >
          <Text style={styles.toggleText}>{!locked ? t('accessDecision.unlocked') : t('accessDecision.locked')}</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

// ─── Main Screen ─────────────────────────────────────────────────────────────

export default function AccessDecision() {
  const { t } = useTranslation();
  return (
    <View style={styles.screen}>
      <StatusBar barStyle="light-content" backgroundColor={COLORS.headerBg} />
      <Header />
      <DropdownBar />
      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        <AccessGrantedBadge t={t} />
        <UserCard t={t} />
        <ConfidenceScore t={t} />
        <DeviceInfo t={t} />
        <ActionButtons t={t} />
        <AudioFeedback t={t} />
        <DoorStatus t={t} />
        <View style={{ height: 32 }} />
      </ScrollView>
    </View>
  );
}

// ─── Styles ──────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: COLORS.bg,
  },

  // Header
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: COLORS.headerBg,
    paddingHorizontal: 16,
    paddingTop: 48,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.cardBorder,
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  logoIcon: {
    color: COLORS.green,
    fontSize: 20,
    marginRight: 6,
  },
  logoText: {
    fontSize: 20,
    fontWeight: '800',
    letterSpacing: 2,
  },
  logoVein: {
    color: COLORS.white,
  },
  logoGuard: {
    color: COLORS.green,
  },
  avatarCircle: {
    width: 38,
    height: 38,
    borderRadius: 19,
    backgroundColor: COLORS.cardBorder,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1.5,
    borderColor: COLORS.teal,
  },
  avatarText: {
    fontSize: 18,
  },

  // Dropdown
  dropdown: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginHorizontal: 16,
    marginTop: 10,
    marginBottom: 6,
    paddingVertical: 10,
    paddingHorizontal: 14,
    backgroundColor: COLORS.cardBg,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: COLORS.cardBorder,
  },
  dropdownText: {
    color: COLORS.text,
    fontSize: 13,
  },
  dropdownArrow: {
    color: COLORS.textDim,
    fontSize: 10,
  },

  // Scroll
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    paddingHorizontal: 16,
    paddingTop: 16,
  },

  // ACCESS GRANTED Badge
  badgeContainer: {
    alignItems: 'center',
    paddingVertical: 28,
    backgroundColor: COLORS.cardBg,
    borderRadius: 12,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: COLORS.cardBorder,
  },
  checkCircleOuter: {
    width: 90,
    height: 90,
    borderRadius: 45,
    borderWidth: 3,
    borderColor: COLORS.green,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 16,
    shadowColor: COLORS.green,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 12,
    elevation: 8,
  },
  checkCircleInner: {
    width: 74,
    height: 74,
    borderRadius: 37,
    backgroundColor: COLORS.greenDark,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 2,
    borderColor: COLORS.green,
  },
  checkMark: {
    color: COLORS.green,
    fontSize: 36,
    fontWeight: '900',
  },
  accessGrantedText: {
    color: COLORS.green,
    fontSize: 28,
    fontWeight: '800',
    letterSpacing: 3,
    textShadowColor: COLORS.green,
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 10,
    marginBottom: 6,
  },
  accessSubText: {
    color: COLORS.textDim,
    fontSize: 13,
  },

  // Card
  card: {
    backgroundColor: COLORS.cardBg,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: COLORS.cardBorder,
    padding: 14,
    marginBottom: 12,
  },

  // User card
  userRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  userAvatar: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: '#1a3552',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
    borderWidth: 2,
    borderColor: COLORS.cardBorder,
  },
  userAvatarText: {
    fontSize: 26,
  },
  userInfo: {
    flex: 1,
  },
  userName: {
    color: COLORS.white,
    fontSize: 16,
    fontWeight: '700',
  },
  userRole: {
    color: COLORS.green,
    fontSize: 12,
    marginTop: 2,
  },
  divider: {
    height: 1,
    backgroundColor: COLORS.cardBorder,
    marginBottom: 10,
  },
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 5,
  },
  detailLabel: {
    color: COLORS.textDim,
    fontSize: 12,
  },
  detailValue: {
    color: COLORS.white,
    fontSize: 12,
    fontWeight: '600',
  },
  greenText: {
    color: COLORS.green,
    fontWeight: '700',
  },

  // Confidence score
  confidenceHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  confidenceValue: {
    color: COLORS.green,
    fontSize: 18,
    fontWeight: '800',
  },
  progressBarBg: {
    height: 10,
    backgroundColor: '#0d2a1a',
    borderRadius: 5,
    overflow: 'hidden',
  },
  progressBarFill: {
    height: '100%',
    backgroundColor: COLORS.green,
    borderRadius: 5,
    shadowColor: COLORS.green,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.9,
    shadowRadius: 6,
  },

  // Device info grid
  deviceGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  deviceCell: {
    width: '50%',
    paddingVertical: 6,
  },
  deviceLabel: {
    color: COLORS.textDim,
    fontSize: 11,
    marginBottom: 2,
  },
  deviceValue: {
    color: COLORS.white,
    fontSize: 13,
    fontWeight: '700',
  },

  // Action buttons
  actionsContainer: {
    marginBottom: 12,
    gap: 8,
  },
  btnViewEvent: {
    backgroundColor: 'transparent',
    borderWidth: 1.5,
    borderColor: COLORS.teal,
    borderRadius: 8,
    paddingVertical: 14,
    alignItems: 'center',
  },
  btnViewEventText: {
    color: COLORS.teal,
    fontSize: 13,
    fontWeight: '700',
    letterSpacing: 1.5,
  },
  btnRetry: {
    backgroundColor: COLORS.amberDark,
    borderWidth: 1.5,
    borderColor: COLORS.amber,
    borderRadius: 8,
    paddingVertical: 14,
    alignItems: 'center',
  },
  btnRetryText: {
    color: COLORS.amber,
    fontSize: 13,
    fontWeight: '700',
    letterSpacing: 1.5,
  },
  btnReport: {
    backgroundColor: COLORS.redDark,
    borderWidth: 1.5,
    borderColor: COLORS.red,
    borderRadius: 8,
    paddingVertical: 14,
    alignItems: 'center',
  },
  btnReportText: {
    color: COLORS.red,
    fontSize: 13,
    fontWeight: '700',
    letterSpacing: 1.5,
  },

  // Section cards (Audio / Door)
  sectionCard: {
    backgroundColor: COLORS.cardBg,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: COLORS.cardBorder,
    padding: 14,
    marginBottom: 12,
  },
  sectionTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 8,
  },
  sectionIcon: {
    fontSize: 16,
    color: COLORS.green,
  },
  sectionTitle: {
    color: COLORS.green,
    fontSize: 13,
    fontWeight: '700',
    letterSpacing: 1.5,
  },
  sectionDivider: {
    height: 1,
    backgroundColor: COLORS.cardBorder,
    marginBottom: 10,
  },
  sectionRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  sectionRowLabel: {
    color: COLORS.text,
    fontSize: 13,
  },
  toggleBtn: {
    paddingHorizontal: 14,
    paddingVertical: 7,
    borderRadius: 6,
    alignItems: 'center',
    justifyContent: 'center',
  },
  toggleOn: {
    backgroundColor: COLORS.greenDark,
    borderWidth: 1,
    borderColor: COLORS.green,
  },
  toggleOff: {
    backgroundColor: '#2a1010',
    borderWidth: 1,
    borderColor: COLORS.red,
  },
  toggleText: {
    color: COLORS.white,
    fontSize: 12,
    fontWeight: '700',
    letterSpacing: 1,
  },
});
