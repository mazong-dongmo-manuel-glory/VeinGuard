import React, { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  TextInput,
  StatusBar,
  Platform,
  FlatList,
  Alert,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { BlurView } from 'expo-blur';
import { useFocusEffect } from '@react-navigation/native';
import { COLORS, GRADIENTS } from '../theme';
import { useMqttStore } from '../store/mqttStore';
import { useAuthStore } from '../store/authStore';

function RoleBadge({ role, color }) {
  return (
    <View style={[styles.badge, { borderColor: `${color}55`, backgroundColor: `${color}12` }]}>
      <Text style={[styles.badgeText, { color }]}>{role}</Text>
    </View>
  );
}

function StatusBadge({ status, color }) {
  return (
    <View style={[styles.badge, { borderColor: `${color}55`, backgroundColor: `${color}12` }]}>
      <Text style={[styles.badgeText, { color }]}>{status}</Text>
    </View>
  );
}

function UserCard({ user, onEdit, onDelete, compact, showTechnicalDetails }) {
  const { t } = useTranslation();
  const roleColor = user.role === 'admin' ? '#2060ff' : COLORS.neonCyan;
  const statusColor = COLORS.neonGreen;

  return (
    <BlurView intensity={15} tint="dark" style={[styles.userCard, compact && styles.userCardCompact]}>
      <View style={styles.userCardHeader}>
        <View style={styles.userAvatar}>
          <LinearGradient colors={['#1c3d5a', '#0d1b2e']} style={styles.avatarInner}>
            <Ionicons name="person" size={24} color={COLORS.neonCyan} />
          </LinearGradient>
        </View>
        <View style={styles.userInfo}>
          <Text numberOfLines={1} style={styles.userName}>{user.username}</Text>
          {showTechnicalDetails ? (
            <Text numberOfLines={1} style={styles.userEmail}>{user.email || user.role.toUpperCase()}</Text>
          ) : null}
          <View style={styles.badgeRow}>
            <RoleBadge role={String(user.role || '').toUpperCase()} color={roleColor} />
            <StatusBadge status={t('userManagement.activeStatus')} color={statusColor} />
          </View>
        </View>
      </View>

      {showTechnicalDetails ? (
        <View style={styles.userDetails}>
          <View style={styles.detailItem}>
            <Text style={styles.detailLabel}>{t('userManagement.enrolledAt').toUpperCase()}</Text>
            <Text style={styles.detailValue}>{new Date(user.created_at).toLocaleDateString()}</Text>
          </View>
          <View style={styles.detailItem}>
            <Text style={styles.detailLabel}>{t('userManagement.idLabel')}</Text>
            <Text style={[styles.detailValue, { color: COLORS.neonCyan }]}>#{user.id}</Text>
          </View>
        </View>
      ) : null}

      <View style={styles.cardActions}>
        <TouchableOpacity style={[styles.actionBtn, { borderColor: COLORS.neonPurple }]} onPress={onEdit}>
          <Ionicons name="create-outline" size={16} color={COLORS.neonPurple} />
          <Text style={[styles.actionBtnText, { color: COLORS.neonPurple }]}>{t('userManagement.editBtn').toUpperCase()}</Text>
        </TouchableOpacity>
        <TouchableOpacity style={[styles.actionBtn, { borderColor: COLORS.neonRed }]} onPress={onDelete}>
          <Ionicons name="trash-outline" size={16} color={COLORS.neonRed} />
          <Text style={[styles.actionBtnText, { color: COLORS.neonRed }]}>{t('userManagement.deleteBtn').toUpperCase()}</Text>
        </TouchableOpacity>
      </View>
    </BlurView>
  );
}

export default function UserManagement({ navigation }) {
  const { t } = useTranslation();
  const [search, setSearch] = useState('');
  const [usersList, setUsersList] = useState([]);
  const [refreshing, setRefreshing] = useState(false);

  const fetchUsers = useMqttStore((state) => state.fetchUsers);
  const deleteUser = useMqttStore((state) => state.deleteUser);
  const isConnected = useMqttStore((state) => state.isConnected);
  const preferences = useAuthStore((state) => state.preferences);
  const autoRefreshData = Boolean(preferences?.autoRefreshData);
  const compactLists = Boolean(preferences?.compactLists);
  const showTechnicalDetails = Boolean(preferences?.showTechnicalDetails);

  const loadUsers = useCallback(async () => {
    if (!isConnected) {
      setUsersList([]);
      return;
    }

    try {
      const list = await fetchUsers();
      setUsersList(Array.isArray(list) ? list : []);
    } catch (err) {
      console.error('Failed to fetch users:', err);
    }
  }, [isConnected, fetchUsers]);

  useFocusEffect(
    useCallback(() => {
      loadUsers();
    }, [loadUsers]),
  );

  useEffect(() => {
    if (!autoRefreshData || !isConnected) {
      return undefined;
    }

    const interval = setInterval(() => {
      loadUsers();
    }, 10000);

    return () => clearInterval(interval);
  }, [autoRefreshData, isConnected, loadUsers]);

  const handleRefresh = async () => {
    setRefreshing(true);
    await loadUsers();
    setRefreshing(false);
  };

  const handleDelete = (user) => {
    Alert.alert(
      'Supprimer utilisateur',
      `Supprimer ${user.username} ?`,
      [
        { text: t('common.cancel'), style: 'cancel' },
        {
          text: t('common.delete'),
          style: 'destructive',
          onPress: async () => {
            try {
              await deleteUser(user.id);
              await loadUsers();
            } catch (error) {
              Alert.alert(t('common.error'), error?.message || 'Suppression impossible.');
            }
          },
        },
      ],
    );
  };

  const filteredUsers = usersList.filter((user) => {
    const haystack = `${user.username || ''} ${user.email || ''} ${user.role || ''}`.toLowerCase();
    return haystack.includes(search.toLowerCase());
  });

  return (
    <View style={styles.screen}>
      <StatusBar barStyle="light-content" />
      <LinearGradient colors={GRADIENTS.primary} style={StyleSheet.absoluteFill} />

      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation?.goBack()} style={styles.backBtn}>
          <Ionicons name="chevron-back" size={24} color={COLORS.white} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>{t('userManagement.title')}</Text>
        <TouchableOpacity style={styles.addBtn} onPress={() => navigation?.navigate('EnrollUser', { mode: 'create' })}>
          <LinearGradient colors={GRADIENTS.neonCyan} style={styles.addBtnGradient}>
            <Ionicons name="add" size={24} color={COLORS.white} />
          </LinearGradient>
        </TouchableOpacity>
      </View>

      <FlatList
        data={filteredUsers}
        keyExtractor={(item, index) => String(item.id || index)}
        refreshing={refreshing}
        onRefresh={handleRefresh}
        renderItem={({ item }) => (
          <UserCard
            user={item}
            onEdit={() => navigation?.navigate('EnrollUser', { mode: 'edit', user: item })}
            onDelete={() => handleDelete(item)}
            compact={compactLists}
            showTechnicalDetails={showTechnicalDetails}
          />
        )}
        extraData={{ compactLists, showTechnicalDetails }}
        ListHeaderComponent={(
          <View>
            <View style={styles.welcomeSection}>
              <Text style={styles.subtitle}>{t('userManagement.subtitle')}</Text>
            </View>

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

            <View style={styles.listHeader}>
              <Text style={styles.listTitle}>{t('userManagement.activeStream')}</Text>
              <Text style={styles.listCount}>{t('userManagement.onlineCount', { count: filteredUsers.length })}</Text>
            </View>
          </View>
        )}
        ListEmptyComponent={(
          <Text style={styles.emptyText}>
            {isConnected ? t('accessHistory.noEvents') : t('userManagement.gatewayDisconnected')}
          </Text>
        )}
        contentContainerStyle={styles.scrollContent}
        style={styles.scroll}
        showsVerticalScrollIndicator={false}
      />
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
  scrollContent: { paddingTop: 10, paddingBottom: 40 },
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
  },
  searchInput: { flex: 1, color: COLORS.white, paddingVertical: 15, marginLeft: 10, fontSize: 15 },
  listHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: 15, gap: 12 },
  listTitle: { color: COLORS.textDim, fontSize: 11, fontWeight: '800', letterSpacing: 2 },
  listCount: { color: COLORS.neonCyan, fontSize: 10, fontWeight: '700', textAlign: 'right' },
  emptyText: { color: COLORS.textDim, textAlign: 'center', marginTop: 20 },
  userCard: {
    borderRadius: 24,
    padding: 20,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.05)',
    marginBottom: 15,
    overflow: 'hidden',
  },
  userCardCompact: {
    paddingVertical: 16,
    paddingHorizontal: 16,
  },
  userCardHeader: { flexDirection: 'row', alignItems: 'flex-start', marginBottom: 20 },
  userAvatar: { width: 50, height: 50, borderRadius: 25, overflow: 'hidden', marginRight: 15 },
  avatarInner: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  userInfo: { flex: 1, minWidth: 0 },
  userName: { color: COLORS.white, fontSize: 17, fontWeight: '800' },
  userEmail: { color: COLORS.textDim, fontSize: 13, marginTop: 2, marginBottom: 8 },
  badgeRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  badge: { paddingHorizontal: 10, paddingVertical: 4, borderRadius: 8, borderWidth: 1 },
  badgeText: { fontSize: 10, fontWeight: '900', letterSpacing: 0.5 },
  userDetails: {
    flexDirection: 'row',
    gap: 12,
    backgroundColor: 'rgba(255, 255, 255, 0.03)',
    borderRadius: 16,
    padding: 15,
    marginBottom: 20,
  },
  detailItem: { flex: 1, minWidth: 0 },
  detailLabel: { color: COLORS.textDim, fontSize: 9, fontWeight: '800', letterSpacing: 1, marginBottom: 4 },
  detailValue: { color: COLORS.white, fontSize: 12, fontWeight: '700', flexShrink: 1 },
  cardActions: { flexDirection: 'row', gap: 10 },
  actionBtn: {
    flex: 1,
    paddingVertical: 12,
    borderRadius: 12,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
    gap: 8,
  },
  actionBtnText: { fontSize: 11, fontWeight: '900', letterSpacing: 1 },
});
