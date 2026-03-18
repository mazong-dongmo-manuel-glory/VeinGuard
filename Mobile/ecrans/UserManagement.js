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
  text: '#b8cfe0',
  textDim: '#4a6a8a',
  white: '#ffffff',
  headerBg: '#0a1525',
  inputBg: '#091525',
  greenDark: '#002a1a',
  redDark: '#2a0010',
  amberDark: '#2a1500',
};

const users = [
  {
    name: 'John Mitchell', email: 'john.mitchell@corp.com',
    role: 'ADMIN', roleColor: '#2060ff', status: 'ACTIVE', statusColor: COLORS.green,
    lastAccess: '2 hours ago', device: 'ESP32-01', enrollments: '3 devices',
    actions: ['edit', 'suspend'],
  },
  {
    name: 'Sarah Chen', email: 'sarah.chen@corp.com',
    role: 'OPERATOR', roleColor: '#7000cc', status: 'ACTIVE', statusColor: COLORS.green,
    lastAccess: '1 day ago', device: 'ESP32-02', enrollments: '2 devices',
    actions: ['edit', 'suspend'],
  },
  {
    name: 'Mike Rodriguez', email: 'mike.rodriguez@corp.com',
    role: 'USER', roleColor: COLORS.teal, status: 'SUSPENDED', statusColor: COLORS.amber,
    lastAccess: '5 days ago', device: 'None', enrollments: '1 device',
    actions: ['activate', 'delete'],
  },
  {
    name: 'Emma Thompson', email: 'emma.thompson@corp.com',
    role: 'OPERATOR', roleColor: '#7000cc', status: 'ACTIVE', statusColor: COLORS.green,
    lastAccess: '30 min ago', device: 'ESP32-03', enrollments: '1 device',
    actions: ['edit', 'suspend'],
  },
  {
    name: 'David Park', email: 'david.park@corp.com',
    role: 'OPERATOR', roleColor: '#7000cc', status: 'ACTIVE', statusColor: COLORS.green,
    lastAccess: '4 hours ago', device: 'ESP32-01', enrollments: '2 devices',
    actions: ['edit', 'suspend'],
  },
  {
    name: 'Lisa Wong', email: 'lisa.wong@corp.com',
    role: 'USER', roleColor: COLORS.teal, status: 'ACTIVE', statusColor: COLORS.green,
    lastAccess: '1 hour ago', device: 'ESP32-02', enrollments: '1 device',
    actions: ['edit', 'suspend'],
  },
];

function RoleBadge({ role, color }) {
  return (
    <View style={[styles.badge, { borderColor: color, backgroundColor: color + '22' }]}>
      <Text style={[styles.badgeText, { color }]}>{role}</Text>
    </View>
  );
}

function StatusBadge({ status, color }) {
  return (
    <View style={[styles.badge, { borderColor: color, backgroundColor: color + '22', marginLeft: 4 }]}>
      <Text style={[styles.badgeText, { color }]}>{status}</Text>
    </View>
  );
}

function UserCard({ user }) {
  const { t } = useTranslation();
  const isActivate = user.actions.includes('activate');
  const isDelete = user.actions.includes('delete');
  return (
    <View style={styles.userCard}>
      <View style={styles.userCardHeader}>
        <View style={styles.userCardLeft}>
          <View style={styles.userAvatar}><Text style={styles.avatarText}>👤</Text></View>
          <View style={styles.userInfo}>
            <Text style={styles.userName}>{user.name}</Text>
            <Text style={styles.userEmail}>{user.email}</Text>
            <View style={styles.badgeRow}>
              <RoleBadge role={user.role} color={user.roleColor} />
              <StatusBadge status={user.status} color={user.statusColor} />
            </View>
          </View>
        </View>
        <View style={styles.userCardRight}>
          <View style={styles.checkboxEmpty} />
        </View>
      </View>
      <View style={styles.userDetails}>
        <View style={styles.detailRow}>
          <Text style={styles.detailLabel}>{t('userManagement.lastAccess')}</Text>
          <Text style={styles.detailValue}>{user.lastAccess}</Text>
        </View>
        <View style={styles.detailRow}>
          <Text style={styles.detailLabel}>{t('userManagement.device')}</Text>
          <Text style={[styles.detailValue, { color: COLORS.teal }]}>{user.device}</Text>
        </View>
        <View style={styles.detailRow}>
          <Text style={styles.detailLabel}>{t('userManagement.enrollments')}</Text>
          <Text style={styles.detailValue}>{user.enrollments}</Text>
        </View>
      </View>
      <View style={styles.actionRow}>
        {isActivate ? (
          <>
            <TouchableOpacity style={[styles.actionBtn, styles.activateBtn]}>
              <Text style={[styles.actionBtnText, { color: COLORS.green }]}>{t('userManagement.activateBtn')}</Text>
            </TouchableOpacity>
            <TouchableOpacity style={[styles.actionBtn, styles.deleteBtn]}>
              <Text style={[styles.actionBtnText, { color: COLORS.red }]}>{t('userManagement.deleteBtn')}</Text>
            </TouchableOpacity>
          </>
        ) : (
          <>
            <TouchableOpacity style={[styles.actionBtn, styles.editBtn]}>
              <Text style={[styles.actionBtnText, { color: COLORS.green }]}>{t('userManagement.editBtn')}</Text>
            </TouchableOpacity>
            <TouchableOpacity style={[styles.actionBtn, styles.suspendBtn]}>
              <Text style={[styles.actionBtnText, { color: COLORS.amber }]}>{t('userManagement.suspendBtn')}</Text>
            </TouchableOpacity>
          </>
        )}
      </View>
    </View>
  );
}

export default function UserManagement({ navigation }) {
  const { t } = useTranslation();
  const [search, setSearch] = useState('');

  return (
    <View style={styles.screen}>
      <StatusBar barStyle="light-content" backgroundColor={COLORS.headerBg} />
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.headerLeft}>
          <Text style={styles.logoVein}>VEIN</Text>
          <Text style={styles.logoGuard}>GUARD</Text>
          <View style={styles.mqttBadge}>
            <View style={styles.mqttDot} />
            <Text style={styles.mqttText}>{t('login.mqttBadge')}</Text>
          </View>
        </View>
        <View style={styles.headerRight}>
          <Text style={styles.headerTime}>15:04:00   UTC+0</Text>
          <View style={styles.avatarCircle}><Text>👤</Text></View>
          <Text style={styles.adminText}>{t('common.admin')}</Text>
        </View>
      </View>
      <View style={styles.dropdown}>
        <Text style={styles.dropdownText}>{t('userManagement.dropdownLabel')}</Text>
        <Text style={styles.dropdownArrow}>▼</Text>
      </View>

      <ScrollView style={styles.scroll} showsVerticalScrollIndicator={false}>
        {/* Title row */}
        <View style={styles.titleRow}>
          <View>
            <Text style={styles.pageTitle}>{t('userManagement.title')}</Text>
            <Text style={styles.pageSubtitle}>{t('userManagement.subtitle')}</Text>
          </View>
          <TouchableOpacity
            style={styles.enrollBtn}
            onPress={() => navigation?.navigate('EnrollUser')}
            activeOpacity={0.85}
          >
              <Text style={styles.enrollBtnText}>{t('userManagement.addUser')}</Text>
          </TouchableOpacity>
        </View>

        {/* Search & filters */}
        <View style={styles.filtersCard}>
          <View style={styles.searchRow}>
            <Text style={styles.searchIcon}>🔍</Text>
            <TextInput
              style={styles.searchInput}
              placeholder={t('userManagement.searchPlaceholder')}
              placeholderTextColor={COLORS.textDim}
              value={search}
              onChangeText={setSearch}
            />
          </View>
          <View style={styles.filterRow}>
            <View style={styles.filterSelect}>
              <Text style={styles.filterText}>{t('userManagement.allRoles')}</Text>
              <Text style={styles.filterArrow}>▼</Text>
            </View>
          </View>
          <View style={styles.filterRow}>
            <View style={styles.filterSelect}>
              <Text style={styles.filterText}>{t('userManagement.allStatus')}</Text>
              <Text style={styles.filterArrow}>▼</Text>
            </View>
            <TouchableOpacity style={styles.bulkDisableBtn}>
              <Text style={styles.bulkDisableBtnText}>{t('userManagement.bulkDisable')}</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.firstChoiceBtn}>
              <Text style={styles.firstChoiceBtnText}>{t('userManagement.firstChoice')}</Text>
            </TouchableOpacity>
          </View>
        </View>

        {/* User list */}
        {users.map((user, i) => (
          <UserCard key={i} user={user} />
        ))}

        <View style={{ height: 32 }} />
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: COLORS.bg },
  header: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    backgroundColor: COLORS.headerBg, paddingHorizontal: 16, paddingTop: 44, paddingBottom: 10,
    borderBottomWidth: 1, borderBottomColor: COLORS.cardBorder,
  },
  headerLeft: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  logoVein: { color: '#fff', fontWeight: '900', fontSize: 17 },
  logoGuard: { color: COLORS.teal, fontWeight: '900', fontSize: 17, marginRight: 10 },
  mqttBadge: { flexDirection: 'row', alignItems: 'center', borderWidth: 1, borderColor: COLORS.green, borderRadius: 20, paddingHorizontal: 8, paddingVertical: 3 },
  mqttDot: { width: 7, height: 7, borderRadius: 3.5, backgroundColor: COLORS.green, marginRight: 4 },
  mqttText: { color: COLORS.green, fontSize: 9, fontWeight: '700' },
  headerRight: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  headerTime: { color: COLORS.textDim, fontSize: 10 },
  avatarCircle: { width: 28, height: 28, borderRadius: 14, backgroundColor: COLORS.cardBorder, alignItems: 'center', justifyContent: 'center' },
  adminText: { color: COLORS.text, fontSize: 11 },
  dropdown: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    marginHorizontal: 14, marginVertical: 8, padding: 10,
    backgroundColor: COLORS.cardBg, borderRadius: 8, borderWidth: 1, borderColor: COLORS.cardBorder,
  },
  dropdownText: { color: COLORS.teal, fontSize: 12 },
  dropdownArrow: { color: COLORS.textDim, fontSize: 10 },
  scroll: { flex: 1, paddingHorizontal: 14 },
  titleRow: { flexDirection: 'row', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 12, marginTop: 4 },
  pageTitle: { color: COLORS.white, fontSize: 20, fontWeight: '900', letterSpacing: 2 },
  pageSubtitle: { color: COLORS.textDim, fontSize: 10, marginTop: 2 },
  enrollBtn: { backgroundColor: COLORS.greenDark, borderWidth: 1, borderColor: COLORS.green, borderRadius: 8, paddingHorizontal: 12, paddingVertical: 8, marginTop: 4 },
  enrollBtnText: { color: COLORS.green, fontSize: 11, fontWeight: '800', letterSpacing: 0.5 },
  filtersCard: { backgroundColor: COLORS.cardBg, borderRadius: 10, borderWidth: 1, borderColor: COLORS.cardBorder, padding: 12, marginBottom: 12 },
  searchRow: { flexDirection: 'row', alignItems: 'center', backgroundColor: COLORS.inputBg, borderWidth: 1, borderColor: COLORS.cardBorder, borderRadius: 6, paddingHorizontal: 10, marginBottom: 10 },
  searchIcon: { fontSize: 13, marginRight: 6 },
  searchInput: { flex: 1, color: COLORS.white, paddingVertical: 10, fontSize: 13 },
  filterRow: { flexDirection: 'row', gap: 8, marginBottom: 8 },
  filterSelect: { flex: 1, flexDirection: 'row', justifyContent: 'space-between', backgroundColor: COLORS.inputBg, borderWidth: 1, borderColor: COLORS.cardBorder, borderRadius: 6, paddingHorizontal: 10, paddingVertical: 9 },
  filterText: { color: COLORS.text, fontSize: 12 },
  filterArrow: { color: COLORS.textDim, fontSize: 9 },
  bulkDisableBtn: { borderWidth: 1, borderColor: COLORS.amber, borderRadius: 6, paddingHorizontal: 10, paddingVertical: 9 },
  bulkDisableBtnText: { color: COLORS.amber, fontSize: 10, fontWeight: '700' },
  firstChoiceBtn: { borderWidth: 1, borderColor: '#c000ff', borderRadius: 6, paddingHorizontal: 10, paddingVertical: 9 },
  firstChoiceBtnText: { color: '#c000ff', fontSize: 10, fontWeight: '700' },
  userCard: { backgroundColor: COLORS.cardBg, borderRadius: 10, borderWidth: 1, borderColor: COLORS.cardBorder, padding: 14, marginBottom: 10 },
  userCardHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 10 },
  userCardLeft: { flexDirection: 'row', alignItems: 'flex-start', flex: 1 },
  userAvatar: { width: 44, height: 44, borderRadius: 22, backgroundColor: '#1a2e48', alignItems: 'center', justifyContent: 'center', marginRight: 10 },
  avatarText: { fontSize: 22 },
  userInfo: { flex: 1 },
  userName: { color: COLORS.white, fontSize: 14, fontWeight: '700' },
  userEmail: { color: COLORS.textDim, fontSize: 10, marginBottom: 5 },
  badgeRow: { flexDirection: 'row', flexWrap: 'wrap' },
  badge: { paddingHorizontal: 7, paddingVertical: 2, borderRadius: 4, borderWidth: 1, marginRight: 4, marginBottom: 2 },
  badgeText: { fontSize: 9, fontWeight: '800', letterSpacing: 0.5 },
  userCardRight: {},
  checkboxEmpty: { width: 18, height: 18, borderRadius: 3, borderWidth: 1.5, borderColor: COLORS.cardBorder },
  userDetails: { marginBottom: 10 },
  detailRow: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 3 },
  detailLabel: { color: COLORS.textDim, fontSize: 11 },
  detailValue: { color: COLORS.white, fontSize: 11, fontWeight: '600' },
  actionRow: { flexDirection: 'row', gap: 8 },
  actionBtn: { flex: 1, paddingVertical: 10, borderRadius: 7, alignItems: 'center', borderWidth: 1 },
  editBtn: { backgroundColor: COLORS.greenDark, borderColor: COLORS.green },
  suspendBtn: { backgroundColor: COLORS.amberDark, borderColor: COLORS.amber },
  activateBtn: { backgroundColor: COLORS.greenDark, borderColor: COLORS.green },
  deleteBtn: { backgroundColor: COLORS.redDark, borderColor: COLORS.red },
  actionBtnText: { fontSize: 11, fontWeight: '800', letterSpacing: 1 },
});
