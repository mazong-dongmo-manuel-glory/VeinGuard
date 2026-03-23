// Centralized Design Tokens for VeinGuard Mobile Application
// This allows for global theme switching and prevents duplicated color definitions across screens.

export const COLORS = {
  // Primary Palette (Deep Space)
  bg: '#050a14',
  cardBg: 'rgba(13, 27, 46, 0.8)', // Glassmorphism base
  cardBorder: '#1c3d5a',
  headerBg: '#07111d',
  inputBg: '#030812',

  // Neon Accents
  neonCyan: '#00f2ff',
  neonGreen: '#39ff14',
  neonAmber: '#ffaa00',
  neonRed: '#ff3d5a',
  neonPurple: '#bc13fe',
  neonMagenta: '#ff00ff',

  // Functional Colors (Semantic)
  success: '#39ff14',
  error: '#ff3d5a',
  warning: '#ffaa00',
  info: '#00f2ff',

  // Typography
  textPrimary: '#ffffff',
  textSecondary: '#a0aec0',
  textDim: '#4a5568',
  
  // Legacy mapping (to avoid breaking existing imports until fully refactored)
  teal: '#00f2ff',
  green: '#39ff14',
  amber: '#ffaa00',
  red: '#ff3d5a',
  purple: '#bc13fe',
  magenta: '#ff00ff',
  text: '#ffffff',
  white: '#ffffff',
  borderBlue: '#1c3d5a',
};

export const GRADIENTS = {
  primary: ['#000b18', '#050a14'],
  neonCyan: ['#00f2ff', '#0090ff'],
  neonPurple: ['#bc13fe', '#8a2be2'],
  neonGreen: ['#39ff14', '#008f11'],
};

export const SHADOWS = {
  cyan: {
    shadowColor: '#00f2ff',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.5,
    shadowRadius: 10,
    elevation: 5,
  },
  green: {
    shadowColor: '#39ff14',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.5,
    shadowRadius: 10,
    elevation: 5,
  },
};
