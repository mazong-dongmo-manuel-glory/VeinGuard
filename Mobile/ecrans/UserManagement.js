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

import { useMqttStore } from '../store/mqttStore';
import { useEffect } from 'react';

function UserCard({ user }) {
  const { t } = useTranslation();
  
  // Mapping backend data to UI
  const roleColor = user.role === 'admin' ? '#2060ff' : COLORS.neonCyan;
  const statusColor = COLORS.neonGreen; // Default to active for now
  
  return (
    <BlurView intensity={15} tint="dark" style={styles.userCard}>
      <View style={styles.userCardHeader}>
        <View style={styles.userAvatar}>
          <LinearGradient colors={['#1c3d5a', '#0d1b2e']} style={styles.avatarInner}>
            <Ionicons name="person" size={24} color={COLORS.neonCyan} />
          </LinearGradient>
        </View>
        <View style={styles.userInfo}>
          <Text style={styles.userName}>{user.username}</Text>
          <Text style={styles.userEmail}>{user.role.toUpperCase()}</Text>
          <View style={styles.badgeRow}>
            <RoleBadge role={user.role.toUpperCase()} color={roleColor} />
            <StatusBadge status="ACTIVE" color={statusColor} />
          </View>
        </View>
        <TouchableOpacity style={styles.moreBtn}>
          <Ionicons name="ellipsis-vertical" size={20} color={COLORS.textDim} />
        </TouchableOpacity>
      </View>
      
      <View style={styles.userDetails}>
        <View style={styles.detailItem}>
          <Text style={styles.detailLabel}>{t('userManagement.enrolledAt').toUpperCase()}</Text>
          <Text style={styles.detailValue}>{new Date(user.created_at).toLocaleDateString()}</Text>
        </View>
        <View style={styles.detailItem}>
          <Text style={styles.detailLabel}>ID</Text>
          <Text style={[styles.detailValue, { color: COLORS.neonCyan }]}>#00{user.id}</Text>
        </View>
      </View>

      <View style={styles.cardActions}>
        <TouchableOpacity style={[styles.actionBtn, { borderColor: COLORS.neonPurple }]}>
          <Text style={[styles.actionBtnText, { color: COLORS.neonPurple }]}>{t('userManagement.editBtn').toUpperCase()}</Text>
        </TouchableOpacity>
        <TouchableOpacity style={[styles.actionBtn, { borderColor: COLORS.textDim, opacity: 0.6 }]}>
          <Ionicons name="settings-outline" size={18} color={COLORS.textDim} />
        </TouchableOpacity>
      </View>
    </BlurView>
  );
}

export default function UserManagement({ navigation }) {
  const { t } = useTranslation();
  const [search, setSearch] = useState('');
  const [usersList, setUsersList] = useState([]);
  const fetchUsers = useMqttStore((state) => state.fetchUsers);
  const isConnected = useMqttStore((state) => state.isConnected);

  useEffect(() => {
    if (isConnected) {
      const loadUsers = async () => {
        try {
          const list = await fetchUsers();
          setUsersList(Array.isArray(list) ? list : []);
        } catch (err) {
          console.error('Failed to fetch users:', err);
        }
      };
      loadUsers();
    }
  }, [isConnected, fetchUsers]);

  const filteredUsers = usersList.filter(u => 
    u.username.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <View style={styles.screen}>
      <StatusBar barStyle="light-content" />
      <LinearGradient colors={GRADIENTS.primary} style={StyleSheet.absoluteFill} />

      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation?.goBack()} style={styles.backBtn}>
          <Ionicons name="chevron-back" size={24} color={COLORS.white} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>{t('userManagement.title')}</Text>
        <TouchableOpacity style={styles.addBtn} onPress={() => navigation?.navigate('EnrollUser')}>
          <LinearGradient colors={GRADIENTS.neonCyan} style={styles.addBtnGradient}>
            <Ionicons name="add" size={24} color={COLORS.white} />
          </LinearGradient>
        </TouchableOpacity>
      </View>

      <ScrollView style={styles.scroll} showsVerticalScrollIndicator={false}>
        <View style={styles.welcomeSection}>
          <Text style={styles.subtitle}>{t('userManagement.subtitle')}</Text>
        </View>

        {/* Search & Filters */}
        <BlurView intensity={10} style={styles.searchCard}>
          <View style={styles.searchRow}>
            <Ionicons name="search" size={20} color={COLORS.neonCyan} />
            <TextInput
              style={styles.searchInput}
              placeholder={t('userManagement.searchPlaceholder')}
              placeholderTextColor={COLORS.textDim}
              value={search}
              onChangeText={setSearch}
            />
          </View>
        </BlurView>

        {/* User list */}
        <View style={styles.listHeader}>
          <Text style={styles.listTitle}>OPERATIVES STREAM</Text>
          <Text style={styles.listCount}>{filteredUsers.length} ONLINE</Text>
        </View>

        {filteredUsers.length === 0 && !isConnected && (
            <Text style={{ color: COLORS.textDim, textAlign: 'center', marginTop: 20 }}>
                PROTOCOL ERROR: GATEWAY DISCONNECTED
            </Text>
        )}

        {filteredUsers.map((user, i) => (
          <UserCard key={user.id || i} user={user} />
        ))}

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
  headerTitle: { color: COLORS.white, fontSize: 20, fontWeight: '800', letterSpacing: 1 },
  addBtn: { width: 40, height: 40, borderRadius: 12, overflow: 'hidden' },
  addBtnGradient: { flex: 1, justifyContent: 'center', alignItems: 'center' },

  scroll: { flex: 1, paddingHorizontal: 20 },
  welcomeSection: { marginBottom: 25 },
  subtitle: { color: COLORS.textSecondary, fontSize: 14 },

  searchCard: {
    borderRadius: 24,
    padding: 20,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.05)',
    marginBottom: 30,
  },
  searchRow: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: 16,
    paddingHorizontal: 15,
    marginBottom: 15,
  },
  searchInput: { flex: 1, color: COLORS.white, paddingVertical: 15, marginLeft: 10, fontSize: 15 },
  filterRow: { flexDirection: 'row', gap: 10 },
  filterChip: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: 'rgba(255, 255, 255, 0.03)',
    borderRadius: 12,
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.05)',
  },
  filterChipText: { color: COLORS.textSecondary, fontSize: 12, fontWeight: '600' },

  listHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: 15 },
  listTitle: { color: COLORS.textDim, fontSize: 11, fontWeight: '800', letterSpacing: 2 },
  listCount: { color: COLORS.neonCyan, fontSize: 10, fontWeight: '700' },

  userCard: {
    borderRadius: 24,
    padding: 20,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.05)',
    marginBottom: 15,
    overflow: 'hidden',
  },
  userCardHeader: { flexDirection: 'row', alignItems: 'center', marginBottom: 20 },
  userAvatar: { width: 50, height: 50, borderRadius: 25, overflow: 'hidden', marginRight: 15 },
  avatarInner: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  userInfo: { flex: 1 },
  userName: { color: COLORS.white, fontSize: 17, fontWeight: '800' },
  userEmail: { color: COLORS.textDim, fontSize: 13, marginTop: 2, marginBottom: 8 },
  badgeRow: { flexDirection: 'row' },
  badge: { paddingHorizontal: 10, paddingVertical: 4, borderRadius: 8, borderWidth: 1 },
  badgeText: { fontSize: 10, fontWeight: '900', letterSpacing: 0.5 },
  moreBtn: { padding: 5 },

  userDetails: {
    flexDirection: 'row',
    backgroundColor: 'rgba(255, 255, 255, 0.03)',
    borderRadius: 16,
    padding: 15,
    marginBottom: 20,
  },
  detailItem: { flex: 1 },
  detailLabel: { color: COLORS.textDim, fontSize: 9, fontWeight: '800', letterSpacing: 1, marginBottom: 4 },
  detailValue: { color: COLORS.white, fontSize: 12, fontWeight: '700' },

  cardActions: { flexDirection: 'row', gap: 10 },
  actionBtn: {
    flex: 1,
    paddingVertical: 12,
    borderRadius: 12,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  actionBtnText: { fontSize: 11, fontWeight: '900', letterSpacing: 1 },
});
