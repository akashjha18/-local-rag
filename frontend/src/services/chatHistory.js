/**
 * chatHistory.js — Chat History Management
 * Saves and loads conversations from localStorage.
 * Each conversation is a session with messages.
 */

const STORAGE_KEY = 'localrag_chat_history';
const MAX_SESSIONS = 10;        // Keep last 10 conversations
const MAX_MESSAGES = 50;        // Max messages per session

/**
 * Generate a unique session ID
 */
const generateSessionId = () => {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
};

/**
 * Load all chat sessions from localStorage
 */
export const loadChatSessions = () => {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return [];
        return JSON.parse(raw);
    } catch (err) {
        console.error('Failed to load chat history:', err);
        return [];
    }
};

/**
 * Save a complete session list to localStorage
 */
const saveSessions = (sessions) => {
    try {
        // Keep only the most recent MAX_SESSIONS
        const trimmed = sessions.slice(-MAX_SESSIONS);
        localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmed));
    } catch (err) {
        console.error('Failed to save chat history:', err);
    }
};

/**
 * Create a new chat session
 * @returns {Object} New session object
 */
export const createSession = () => ({
    id: generateSessionId(),
    title: 'New Chat',
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    messages: [],
});

/**
 * Save messages to a session
 * @param {string} sessionId
 * @param {Array} messages
 * @param {string} title - Auto-generated from first question
 */
export const saveSession = (sessionId, messages, title) => {
    const sessions = loadChatSessions();
    const existingIdx = sessions.findIndex(s => s.id === sessionId);

    const session = {
        id: sessionId,
        title: title || 'Chat',
        createdAt: existingIdx >= 0
            ? sessions[existingIdx].createdAt
            : new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        // Keep only last MAX_MESSAGES
        messages: messages.slice(-MAX_MESSAGES),
    };

    if (existingIdx >= 0) {
        sessions[existingIdx] = session;
    } else {
        sessions.push(session);
    }

    saveSessions(sessions);
};

/**
 * Delete a specific session
 * @param {string} sessionId
 */
export const deleteSession = (sessionId) => {
    const sessions = loadChatSessions();
    const filtered = sessions.filter(s => s.id !== sessionId);
    saveSessions(filtered);
};

/**
 * Clear all chat history
 */
export const clearAllHistory = () => {
    localStorage.removeItem(STORAGE_KEY);
};

/**
 * Generate a session title from the first user message
 * Truncates to 40 chars
 */
export const generateTitle = (firstMessage) => {
    if (!firstMessage) return 'New Chat';
    const text = firstMessage.content || firstMessage;
    return text.length > 40 ? text.substring(0, 40) + '...' : text;
};