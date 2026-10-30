import React, {useEffect, useRef, useState} from 'react';
import {
  Alert,
  BackHandler,
  FlatList,
  Modal,
  Pressable,
  SafeAreaView,
  StyleSheet,
  Switch,
  Text,
  TextInput,
  View,
} from 'react-native';

import AsyncStorage from '@react-native-async-storage/async-storage';
import * as Keychain from 'react-native-keychain';

import {
  LLAMA3_2_1B_SPINQUANT,
  models,
  useLLM,
  useTextEmbeddings,
} from 'react-native-executorch';

import {
  keepLocalCopy,
  pick,
  types,
} from '@react-native-documents/picker';

import {extractText} from 'react-native-pdf-text-extractor';
import {
  DocumentDirectoryPath,
  readFile,
  unlink,
  writeFile,
} from '@dr.pogodin/react-native-fs';

import {open} from 'react-native-nitro-sqlite';

import {
  createVectorTable,
  isVecAvailable,
  knnSearch,
} from 'react-native-nitro-sqlite-vec';

// DOCX files are ZIP containers. JSZip lets React Native read the
// Word XML directly without Node/browser-specific Mammoth builds.
// eslint-disable-next-line @typescript-eslint/no-var-requires
const JSZip = require('jszip');

type Screen = 'chat' | 'documents' | 'settings' | 'account';

type Profile = {
  name: string;
  username: string;
};

type Message = {
  id: string;
  sender: 'user' | 'elie';
  text: string;
  createdAt: number;
};

type DocumentStatus = 'processing' | 'ready' | 'empty' | 'error';
type IndexStatus = 'pending' | 'indexing' | 'indexed' | 'error';

type DocumentFile = {
  id: string;
  name: string;
  uri: string;
  size?: number | null;
  type?: string | null;
  addedAt: number;
  textPath?: string;
  status: DocumentStatus;
  indexStatus: IndexStatus;
};

type Chat = {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
  documentIds: string[];
};

type RagChunk = {
  id: number;
  document_id: string;
  content: string;
  chunk_index: number;
};

type VectorMatch = {
  rowid: number;
  distance: number;
};

const PROFILE_KEY = '@elie_profile';
const SESSION_KEY = '@elie_session';
const CHATS_KEY = '@elie_chats';
const DOCUMENTS_KEY = '@elie_documents';
const KEYCHAIN_SERVICE = 'com.elie.local';

const RAG_DB_NAME = 'elie_rag_minilm.sqlite';
const RAG_VECTOR_TABLE = 'rag_vectors';
const RAG_CHUNK_TABLE = 'rag_chunks';
const EMBEDDING_DIMENSIONS = 384;
const CHUNK_SIZE = 900;
const CHUNK_OVERLAP = 120;
const RETRIEVAL_K = 50;
const CONTEXT_CHUNKS = 5;
const FALLBACK_CONTEXT_CHARS = 18000;

const ragDb = open({name: RAG_DB_NAME});

const createWelcomeChat = (): Chat => {
  const now = Date.now();
  return {
    id: 'welcome',
    title: 'Welcome to Elie',
    messages: [
      {
        id: 'welcome-message',
        sender: 'elie',
        text: 'Hello! I am Elie, your private offline AI assistant.',
        createdAt: now,
      },
    ],
    createdAt: now,
    updatedAt: now,
    documentIds: [],
  };
};

const normalizePath = (uri: string) =>
  uri.startsWith('file://') ? uri.slice(7) : uri;

const isPdfDocument = (name: string, type?: string | null) =>
  type === 'application/pdf' || name.toLowerCase().endsWith('.pdf');

const isDocxDocument = (name: string, type?: string | null) =>
  type ===
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document' ||
  name.toLowerCase().endsWith('.docx');

const splitIntoChunks = (text: string) => {
  const cleaned = text
    .replace(/\r/g, '\n')
    .replace(/[ \t]+/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .trim();

  if (!cleaned) return [];
  if (cleaned.length <= CHUNK_SIZE) return [cleaned];

  const chunks: string[] = [];
  let start = 0;

  while (start < cleaned.length) {
    let end = Math.min(start + CHUNK_SIZE, cleaned.length);

    if (end < cleaned.length) {
      const paragraphBreak = cleaned.lastIndexOf('\n\n', end);
      const sentenceBreak = cleaned.lastIndexOf('. ', end);
      const spaceBreak = cleaned.lastIndexOf(' ', end);
      const preferred = Math.max(paragraphBreak, sentenceBreak, spaceBreak);
      if (preferred > start + CHUNK_SIZE * 0.6) {
        end = preferred + (preferred === sentenceBreak ? 1 : 0);
      }
    }

    const chunk = cleaned.slice(start, end).trim();
    if (chunk) chunks.push(chunk);

    if (end >= cleaned.length) break;
    start = Math.max(end - CHUNK_OVERLAP, start + 1);
  }

  return chunks;
};

export default function App() {
  const [profile, setProfile] = useState<Profile | null>(null);
  const [loggedIn, setLoggedIn] = useState(false);
  const [authMode, setAuthMode] = useState<'login' | 'create'>('login');
  const [name, setName] = useState('');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');

  const [screen, setScreen] = useState<Screen>('chat');
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [chats, setChats] = useState<Chat[]>([]);
  const [activeChatId, setActiveChatId] = useState('');
  const [message, setMessage] = useState('');
  const [documents, setDocuments] = useState<DocumentFile[]>([]);
  const [darkMode, setDarkMode] = useState(true);

  const [chatMenuVisible, setChatMenuVisible] = useState(false);
  const [selectedChat, setSelectedChat] = useState<Chat | null>(null);
  const [renameVisible, setRenameVisible] = useState(false);
  const [renameValue, setRenameValue] = useState('');

  const [attachmentMenuVisible, setAttachmentMenuVisible] = useState(false);
  const [documentViewerVisible, setDocumentViewerVisible] = useState(false);
  const [documentViewerTitle, setDocumentViewerTitle] = useState('');
  const [documentViewerText, setDocumentViewerText] = useState('');

  const [ragReady, setRagReady] = useState(false);
  const [ragError, setRagError] = useState('');

  const indexingQueueRunning = useRef(false);
  const [indexQueueTick, setIndexQueueTick] = useState(0);

  const llm = useLLM({model: LLAMA3_2_1B_SPINQUANT});
  const embeddings = useTextEmbeddings({
    model: models.text_embedding.all_minilm_l6_v2(),
  });

  useEffect(() => {
    loadData();
    initializeRagDatabase();
  }, []);

  useEffect(() => {
    if (!ragReady || !embeddings.isReady || documents.length === 0) return;
    if (indexingQueueRunning.current) return;

    const pending = documents.filter(
      doc => doc.status === 'ready' && doc.indexStatus === 'pending',
    );

    if (pending.length === 0) return;

    processPendingDocuments(pending);
  }, [ragReady, embeddings.isReady, documents.length, indexQueueTick]);

  useEffect(() => {
    const onBackPress = () => {
      if (documentViewerVisible) {
        setDocumentViewerVisible(false);
        return true;
      }
      if (attachmentMenuVisible) {
        setAttachmentMenuVisible(false);
        return true;
      }
      if (chatMenuVisible) {
        setChatMenuVisible(false);
        return true;
      }
      if (renameVisible) {
        setRenameVisible(false);
        return true;
      }
      if (drawerOpen) {
        setDrawerOpen(false);
        return true;
      }
      if (screen !== 'chat') {
        setScreen('chat');
        return true;
      }
      return false;
    };

    const subscription = BackHandler.addEventListener(
      'hardwareBackPress',
      onBackPress,
    );
    return () => subscription.remove();
  }, [
    documentViewerVisible,
    attachmentMenuVisible,
    chatMenuVisible,
    renameVisible,
    drawerOpen,
    screen,
  ]);

  const initializeRagDatabase = () => {
    try {
      ragDb.execute('PRAGMA journal_mode = WAL');
      ragDb.execute('PRAGMA foreign_keys = ON');

      ragDb.execute(`
        CREATE TABLE IF NOT EXISTS ${RAG_CHUNK_TABLE} (
          id INTEGER PRIMARY KEY,
          document_id TEXT NOT NULL,
          content TEXT NOT NULL,
          chunk_index INTEGER NOT NULL
        )
      `);

      if (!isVecAvailable(ragDb)) {
        throw new Error(
          'sqlite-vec is not enabled. Add nitroSqliteVec=true to android/gradle.properties and rebuild.',
        );
      }

      const existing = ragDb.execute<{name: string}>(
        `SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?`,
        [RAG_VECTOR_TABLE],
      ).results;

      if (!existing || existing.length === 0) {
        createVectorTable(ragDb, RAG_VECTOR_TABLE, {
          dimensions: EMBEDDING_DIMENSIONS,
          type: 'float',
          distanceMetric: 'cosine',
          column: 'embedding',
        });
      }

      setRagError('');
      setRagReady(true);
    } catch (error) {
      console.log('RAG DB init error:', error);
      setRagError(String(error));
      setRagReady(false);
    }
  };

  const loadData = async () => {
    try {
      const [savedProfile, savedSession, savedChats, savedDocuments] =
        await Promise.all([
          AsyncStorage.getItem(PROFILE_KEY),
          AsyncStorage.getItem(SESSION_KEY),
          AsyncStorage.getItem(CHATS_KEY),
          AsyncStorage.getItem(DOCUMENTS_KEY),
        ]);

      if (savedProfile) {
        setProfile(JSON.parse(savedProfile));
      } else {
        setAuthMode('create');
      }

      if (savedChats) {
        const rawChats = JSON.parse(savedChats);
        const normalized: Chat[] = rawChats.map((chat: any) => ({
          ...chat,
          documentIds: Array.isArray(chat.documentIds)
            ? chat.documentIds
            : [],
        }));

        if (normalized.length > 0) {
          setChats(normalized);
          setActiveChatId(normalized[0].id);
        } else {
          const welcome = createWelcomeChat();
          setChats([welcome]);
          setActiveChatId(welcome.id);
        }
      } else {
        const welcome = createWelcomeChat();
        setChats([welcome]);
        setActiveChatId(welcome.id);
      }

      if (savedDocuments) {
        const normalizedDocs: DocumentFile[] = JSON.parse(savedDocuments)
          .map((doc: any) => ({
            ...doc,
            status: doc.status ?? (doc.textPath ? 'ready' : 'empty'),
            indexStatus: doc.indexStatus ?? (doc.status === 'ready' ? 'pending' : 'error'),
          }))
          .filter((doc: DocumentFile) =>
            isPdfDocument(doc.name, doc.type) ||
            isDocxDocument(doc.name, doc.type),
          );
        setDocuments(normalizedDocs);
      }

      if (savedSession === 'true' && savedProfile) {
        setLoggedIn(true);
      }
    } catch (error) {
      console.log('Load error:', error);
    }
  };

  const saveChats = async (updatedChats: Chat[]) => {
    setChats(updatedChats);
    try {
      await AsyncStorage.setItem(CHATS_KEY, JSON.stringify(updatedChats));
    } catch (error) {
      console.log('Save chats error:', error);
    }
  };

  const saveDocuments = async (updatedDocuments: DocumentFile[]) => {
    setDocuments(updatedDocuments);
    try {
      await AsyncStorage.setItem(
        DOCUMENTS_KEY,
        JSON.stringify(updatedDocuments),
      );
    } catch (error) {
      console.log('Save documents error:', error);
    }
  };

  const updateDocument = async (
    documentId: string,
    patch: Partial<DocumentFile>,
  ) => {
    const updated = documents.map(doc =>
      doc.id === documentId ? {...doc, ...patch} : doc,
    );
    await saveDocuments(updated);
  };

  const createProfile = async () => {
    if (!name.trim() || !username.trim() || !password) {
      Alert.alert('Missing information', 'Please fill in all fields.');
      return;
    }
    if (password.length < 6) {
      Alert.alert('Password too short', 'Use at least 6 characters.');
      return;
    }

    const newProfile: Profile = {
      name: name.trim(),
      username: username.trim(),
    };

    try {
      await AsyncStorage.setItem(PROFILE_KEY, JSON.stringify(newProfile));
      await Keychain.setGenericPassword(
        newProfile.username,
        password,
        {service: KEYCHAIN_SERVICE},
      );
      await AsyncStorage.setItem(SESSION_KEY, 'true');

      setProfile(newProfile);
      setLoggedIn(true);
      setName('');
      setUsername('');
      setPassword('');
    } catch (error) {
      console.log('Create profile error:', error);
      Alert.alert('Error', 'Could not create profile.');
    }
  };

  const login = async () => {
    if (!username.trim() || !password) {
      Alert.alert('Missing information', 'Enter username and password.');
      return;
    }

    try {
      const credentials = await Keychain.getGenericPassword({
        service: KEYCHAIN_SERVICE,
      });

      if (!credentials) {
        Alert.alert('No profile', 'Create a profile first.');
        setAuthMode('create');
        return;
      }

      if (
        credentials.username !== username.trim() ||
        credentials.password !== password
      ) {
        Alert.alert('Login failed', 'Incorrect username or password.');
        return;
      }

      await AsyncStorage.setItem(SESSION_KEY, 'true');
      setLoggedIn(true);
      setPassword('');
    } catch (error) {
      console.log('Login error:', error);
      Alert.alert('Error', 'Could not access local credentials.');
    }
  };

  const logout = async () => {
    await AsyncStorage.removeItem(SESSION_KEY);
    setLoggedIn(false);
    setDrawerOpen(false);
    setScreen('chat');
    setPassword('');
  };

  const createNewChat = async () => {
    const now = Date.now();
    const newChat: Chat = {
      id: `chat-${now}`,
      title: 'New Chat',
      messages: [],
      createdAt: now,
      updatedAt: now,
      documentIds: [],
    };

    await saveChats([newChat, ...chats]);
    setActiveChatId(newChat.id);
    setScreen('chat');
    setDrawerOpen(false);
  };

  const getChatMessagesForModel = (chat: Chat, currentUserText: string) => {
    const recent = chat.messages.slice(-10).map(item => ({
      role: item.sender === 'user' ? ('user' as const) : ('assistant' as const),
      content: item.text,
    }));

    return [
      {
        role: 'system' as const,
        content:
          'You are Elie, a private offline AI assistant. Be accurate, concise, and helpful. When a user-attached document is provided, treat it as authorized source material and answer questions about its contents directly. Do not refuse merely because the document contains codes, names, identifiers, or codenames. Use the provided document context as the primary source and do not invent facts that are not supported by it.',
      },
      ...recent,
      {
        role: 'user' as const,
        content: currentUserText,
      },
    ];
  };

  const retrieveContext = async (
    query: string,
    allowedDocumentIds: string[],
  ) => {
    if (!ragReady || !embeddings.isReady || allowedDocumentIds.length === 0) {
      return [] as Array<
        RagChunk & {distance: number; sourceName: string}
      >;
    }

    const queryEmbedding = await embeddings.forward(query);
    const queryVector = Array.from(queryEmbedding as Float32Array);

    const matches = knnSearch(
      ragDb,
      RAG_VECTOR_TABLE,
      queryVector,
      RETRIEVAL_K,
    ) as VectorMatch[];

    const results: Array<
      RagChunk & {distance: number; sourceName: string}
    > = [];

    for (const match of matches) {
      const dbResult = ragDb.execute(
        `SELECT id, document_id, content, chunk_index
         FROM ${RAG_CHUNK_TABLE}
         WHERE id = ?`,
        [Number(match.rowid)],
      );

      const row = dbResult.results?.[0] as
        | RagChunk
        | undefined;

      if (!row) continue;

      if (!allowedDocumentIds.includes(String(row.document_id))) {
        continue;
      }

      const source = documents.find(
        doc => doc.id === String(row.document_id),
      );

      const result: RagChunk & {
        distance: number;
        sourceName: string;
      } = {
        id: Number(row.id),
        document_id: String(row.document_id),
        content: String(row.content),
        chunk_index: Number(row.chunk_index),
        distance: Number(match.distance),
        sourceName: source?.name ?? 'Document',
      };

      results.push(result);

      if (results.length >= CONTEXT_CHUNKS) break;
    }

    return results;
  };

  const readAttachedDocumentText = async (
    allowedDocumentIds: string[],
  ) => {
    const attached = documents.filter(
      doc =>
        allowedDocumentIds.includes(doc.id) &&
        doc.status === 'ready' &&
        !!doc.textPath,
    );

    const parts: string[] = [];

    for (const document of attached) {
      try {
        const text = await readFile(
          normalizePath(document.textPath as string),
          'utf8',
        );

        if (text.trim()) {
          parts.push(
            `[Source: ${document.name}]\n${text.trim()}`,
          );
        }
      } catch (error) {
        console.log(
          'Attached document read error:',
          error,
        );
      }
    }

    return parts.join('\n\n').slice(0, FALLBACK_CONTEXT_CHARS);
  };

  const sendMessage = async () => {
    const text = message.trim();
    if (!text || llm.isGenerating) return;

    if (!llm.isReady) {
      Alert.alert('Elie is loading', 'Please wait until Llama 3.2 1B is ready.');
      return;
    }

    const currentChat = chats.find(chat => chat.id === activeChatId) ?? chats[0];
    if (!currentChat) return;

    const now = Date.now();
    const userMessage: Message = {
      id: `message-${now}`,
      sender: 'user',
      text,
      createdAt: now,
    };

    const chatWithUser = {
      ...currentChat,
      title:
        currentChat.messages.length === 0
          ? text.slice(0, 32)
          : currentChat.title,
      messages: [...currentChat.messages, userMessage],
      updatedAt: now,
    };

    const updatedChats = chats.map(chat =>
      chat.id === activeChatId ? chatWithUser : chat,
    );

    await saveChats(updatedChats);
    setMessage('');

    try {
      const attachedIds = Array.isArray(chatWithUser.documentIds)
        ? chatWithUser.documentIds
        : [];

      const retrieved = await retrieveContext(text, attachedIds);

      let modelUserContent = text;

      if (attachedIds.length > 0) {
        const isWholeDocumentRequest =
          /\b(summarize|summarise|summary|overview|what is in|what's in|what is inside)\b/i.test(
            text,
          );

        let context = retrieved
          .map(
            item =>
              `[Source: ${item.sourceName}, chunk ${item.chunk_index + 1}]\n${item.content}`,
          )
          .join('\n\n');

        // For summaries or when vector retrieval returns nothing, use the
        // locally extracted document text as a reliable fallback.
        if (isWholeDocumentRequest || !context.trim()) {
          const fallbackText = await readAttachedDocumentText(
            attachedIds,
          );

          if (fallbackText.trim()) {
            context = fallbackText;
          }
        }

        if (context.trim()) {
          modelUserContent = `The following text comes from a document the user explicitly attached.
Treat this document as user-provided source material, not as confidential system information.
Answer factual questions about it directly, including codes, names, identifiers, and codenames.
Use only the document context when the question is about the document.
If the requested information is not present in the provided context, say that clearly.

DOCUMENT CONTEXT:
${context}

USER QUESTION:
${text}`;
        }
      }

      const response = await llm.generate(
        getChatMessagesForModel(chatWithUser, modelUserContent),
      );

      const aiNow = Date.now();
      const aiMessage: Message = {
        id: `message-${aiNow}-ai`,
        sender: 'elie',
        text: response,
        createdAt: aiNow,
      };

      const finalChats = updatedChats.map(chat =>
        chat.id === activeChatId
          ? {
              ...chat,
              messages: [...chat.messages, aiMessage],
              updatedAt: aiNow,
            }
          : chat,
      );

      await saveChats(finalChats);
    } catch (error) {
      console.log('AI/RAG error:', error);
      Alert.alert(
        'AI Error',
        llm.error ? String(llm.error) : String(error),
      );
    }
  };

  const decodeXmlEntities = (value: string) =>
    value
      .replace(/&amp;/g, '&')
      .replace(/&lt;/g, '<')
      .replace(/&gt;/g, '>')
      .replace(/&quot;/g, '"')
      .replace(/&apos;/g, "'")
      .replace(/&#x([0-9a-fA-F]+);/g, (_, hex: string) =>
        String.fromCodePoint(parseInt(hex, 16)),
      )
      .replace(/&#([0-9]+);/g, (_, code: string) =>
        String.fromCodePoint(parseInt(code, 10)),
      );

  const extractDocxText = async (localPath: string) => {
    const base64 = await readFile(localPath, 'base64');

    const zip = await JSZip.loadAsync(base64, {
      base64: true,
      checkCRC32: true,
    });

    const documentXmlFile = zip.file('word/document.xml');

    if (!documentXmlFile) {
      throw new Error('DOCX does not contain word/document.xml.');
    }

    const documentXml = await documentXmlFile.async('string');

    const text = documentXml
      .replace(/<w:tab\s*\/?>/gi, '\\t')
      .replace(/<w:br\s*\/?>/gi, '\\n')
      .replace(/<\/w:tc>/gi, '\\t')
      .replace(/<\/w:p>/gi, '\\n')
      .replace(/<[^>]+>/g, '')
      .replace(/\\u000B/g, '\\n');

    return decodeXmlEntities(text)
      .replace(/[ \t]+\\n/g, '\\n')
      .replace(/\\n{3,}/g, '\\n\\n')
      .trim();
  };

  const extractDocumentText = async (
    localUri: string,
    fileName: string,
    mimeType?: string | null,
  ) => {
    const localPath = normalizePath(localUri);

    if (isPdfDocument(fileName, mimeType)) {
      return await extractText(localPath, {normalize: true});
    }

    if (isDocxDocument(fileName, mimeType)) {
      return await extractDocxText(localPath);
    }

    throw new Error('Only PDF and DOCX files are supported.');
  };

  const deleteVectorChunksForDocument = (documentId: string) => {
    const rows = ragDb.execute<{id: number}>(
      `SELECT id FROM ${RAG_CHUNK_TABLE} WHERE document_id = ?`,
      [documentId],
    ).results ?? [];

    for (const row of rows) {
      ragDb.execute(
        `DELETE FROM ${RAG_VECTOR_TABLE} WHERE rowid = ?`,
        [row.id],
      );
    }

    ragDb.execute(
      `DELETE FROM ${RAG_CHUNK_TABLE} WHERE document_id = ?`,
      [documentId],
    );
  };

  const indexDocument = async (document: DocumentFile) => {
    if (
      !document.textPath ||
      document.status !== 'ready' ||
      !ragReady ||
      !embeddings.isReady
    ) {
      return;
    }

    try {
      await updateDocument(document.id, {indexStatus: 'indexing'});

      const text = await readFile(
        normalizePath(document.textPath),
        'utf8',
      );

      const chunks = splitIntoChunks(text);

      if (chunks.length === 0) {
        await updateDocument(document.id, {indexStatus: 'error'});
        return;
      }

      deleteVectorChunksForDocument(document.id);

      const maxRow = ragDb.execute<{id: number}>(
        `SELECT COALESCE(MAX(id), 0) + 1 AS id FROM ${RAG_CHUNK_TABLE}`,
      ).results?.[0];

      let nextId = Number(maxRow?.id ?? 1);

      for (let index = 0; index < chunks.length; index += 1) {
        const chunk = chunks[index];
        const embedding = await embeddings.forward(chunk);
        const vector = Array.from(embedding as Float32Array);

        ragDb.execute(
          `INSERT INTO ${RAG_CHUNK_TABLE} (id, document_id, content, chunk_index) VALUES (?, ?, ?, ?)`,
          [nextId, document.id, chunk, index],
        );

        ragDb.execute(
          `INSERT INTO ${RAG_VECTOR_TABLE} (rowid, embedding) VALUES (?, ?)`,
          [nextId, JSON.stringify(vector)],
        );

        nextId += 1;
      }

      await updateDocument(document.id, {indexStatus: 'indexed'});
    } catch (error) {
      console.log('Index document error:', error);
      await updateDocument(document.id, {indexStatus: 'error'});
    }
  };

  const processPendingDocuments = async (
    pendingDocuments: DocumentFile[],
  ) => {
    if (indexingQueueRunning.current) return;
    indexingQueueRunning.current = true;

    try {
      for (const document of pendingDocuments) {
        await indexDocument(document);
      }
    } finally {
      indexingQueueRunning.current = false;
      setIndexQueueTick(tick => tick + 1);
    }
  };

  const addDocument = async () => {
    try {
      const [file] = await pick({
        mode: 'import',
        allowMultiSelection: false,
        type: [types.pdf, types.docx],
      });

      if (!file) return;

      if (!file.hasRequestedType) {
        Alert.alert('Unsupported file', 'Please select a PDF or DOCX file.');
        return;
      }

      const [copyResult] = await keepLocalCopy({
        files: [
          {
            uri: file.uri,
            fileName: file.name ?? 'document',
          },
        ],
        destination: 'documentDirectory',
      });

      if (copyResult.status !== 'success') {
        Alert.alert('Upload failed', 'Could not save the document locally.');
        return;
      }

      const documentId = `doc-${Date.now()}`;
      const localUri = copyResult.localUri;
      const fileName = file.name ?? 'Untitled document';

      const processingDocument: DocumentFile = {
        id: documentId,
        name: fileName,
        uri: localUri,
        size: file.size,
        type: file.type,
        addedAt: Date.now(),
        status: 'processing',
        indexStatus: 'pending',
      };

      await saveDocuments([processingDocument, ...documents]);

      try {
        const extractedText = await extractDocumentText(
          localUri,
          fileName,
          file.type,
        );

        const textPath = `${DocumentDirectoryPath}/${documentId}.txt`;
        await writeFile(textPath, extractedText, 'utf8');

        const readyDocument: DocumentFile = {
          ...processingDocument,
          status: extractedText.trim() ? 'ready' : 'empty',
          indexStatus: extractedText.trim() ? 'pending' : 'error',
          textPath,
        };

        const updatedDocuments = documents.map(doc =>
          doc.id === documentId ? readyDocument : doc,
        );

        await saveDocuments([
          readyDocument,
          ...updatedDocuments.filter(doc => doc.id !== documentId),
        ]);

        if (activeChatId && readyDocument.status === 'ready') {
          const updatedChats = chats.map(chat =>
            chat.id === activeChatId
              ? {
                  ...chat,
                  documentIds: Array.from(
                    new Set([...chat.documentIds, documentId]),
                  ),
                  updatedAt: Date.now(),
                }
              : chat,
          );
          await saveChats(updatedChats);
        }

        setAttachmentMenuVisible(false);

        if (readyDocument.status === 'ready') {
          Alert.alert(
            'Document ready',
            `${fileName} was uploaded and its text was extracted. RAG indexing will continue on the device.`,
          );
        } else {
          Alert.alert(
            'Document saved',
            `${fileName} was saved, but no readable text was found.`,
          );
        }
      } catch (error) {
        console.log('Document extraction error:', error);

        const failedDocument: DocumentFile = {
          ...processingDocument,
          status: 'error',
          indexStatus: 'error',
        };

        const updatedDocuments = documents.map(doc =>
          doc.id === documentId ? failedDocument : doc,
        );

        await saveDocuments([
          failedDocument,
          ...updatedDocuments.filter(doc => doc.id !== documentId),
        ]);

        Alert.alert(
          'Document saved',
          `${fileName} was stored, but its text could not be extracted.`,
        );
      }
    } catch (error: any) {
      console.log('Document picker error:', error);
      const msg = String(error?.message ?? '').toLowerCase();
      if (!msg.includes('cancel')) {
        Alert.alert(
          'Document error',
          error?.message ?? 'Could not import the document.',
        );
      }
    }
  };

  const attachDocument = async (documentId: string) => {
    if (!activeChatId) return;

    const updatedChats = chats.map(chat =>
      chat.id === activeChatId
        ? {
            ...chat,
            documentIds: Array.from(
              new Set([...chat.documentIds, documentId]),
            ),
            updatedAt: Date.now(),
          }
        : chat,
    );

    await saveChats(updatedChats);
    setAttachmentMenuVisible(false);
  };

  const removeDocument = (document: DocumentFile) => {
    Alert.alert(
      'Remove Document?',
      `${document.name} will be removed from Elie.`,
      [
        {text: 'Cancel', style: 'cancel'},
        {
          text: 'Remove',
          style: 'destructive',
          onPress: async () => {
            try {
              deleteVectorChunksForDocument(document.id);
            } catch (error) {
              console.log('Vector delete error:', error);
            }

            try {
              if (document.textPath) {
                await unlink(normalizePath(document.textPath));
              }
            } catch {}

            try {
              await unlink(normalizePath(document.uri));
            } catch {}

            const updatedDocuments = documents.filter(
              item => item.id !== document.id,
            );
            await saveDocuments(updatedDocuments);

            const updatedChats = chats.map(chat => ({
              ...chat,
              documentIds: chat.documentIds.filter(
                id => id !== document.id,
              ),
              updatedAt: Date.now(),
            }));

            await saveChats(updatedChats);
          },
        },
      ],
    );
  };

  const viewDocumentText = async (document: DocumentFile) => {
    if (!document.textPath || document.status !== 'ready') {
      Alert.alert(
        'No text available',
        'This document does not contain extractable text.',
      );
      return;
    }

    try {
      const text = await readFile(
        normalizePath(document.textPath),
        'utf8',
      );

      setDocumentViewerTitle(document.name);
      setDocumentViewerText(text);
      setDocumentViewerVisible(true);
    } catch (error) {
      console.log('View text error:', error);
      Alert.alert('Error', 'Could not read the extracted text.');
    }
  };

  const openChatMenu = (chat: Chat) => {
    setSelectedChat(chat);
    setChatMenuVisible(true);
  };

  const startRename = () => {
    if (!selectedChat) return;
    setRenameValue(selectedChat.title);
    setChatMenuVisible(false);
    setRenameVisible(true);
  };

  const saveRename = async () => {
    if (!selectedChat || !renameValue.trim()) return;

    const updatedChats = chats.map(chat =>
      chat.id === selectedChat.id
        ? {
            ...chat,
            title: renameValue.trim(),
            updatedAt: Date.now(),
          }
        : chat,
    );

    await saveChats(updatedChats);
    setRenameVisible(false);
    setSelectedChat(null);
    setRenameValue('');
  };

  const deleteSelectedChat = () => {
    if (!selectedChat) return;

    setChatMenuVisible(false);

    Alert.alert(
      'Delete Chat?',
      `"${selectedChat.title}" will be removed from this device.`,
      [
        {text: 'Cancel', style: 'cancel'},
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            let updatedChats = chats.filter(
              chat => chat.id !== selectedChat.id,
            );

            if (updatedChats.length === 0) {
              updatedChats = [createWelcomeChat()];
            }

            await saveChats(updatedChats);

            if (selectedChat.id === activeChatId) {
              setActiveChatId(updatedChats[0].id);
            }

            setSelectedChat(null);
          },
        },
      ],
    );
  };

  const formatTime = (timestamp: number) =>
    new Date(timestamp).toLocaleTimeString([], {
      hour: 'numeric',
      minute: '2-digit',
    });

  const activeChat =
    chats.find(chat => chat.id === activeChatId) ?? chats[0];

  const attachedDocuments = documents.filter(document =>
    activeChat?.documentIds?.includes(document.id),
  );

  if (!loggedIn || !profile) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.authContainer}>
          <View style={styles.logo}>
            <Text style={styles.logoText}>E</Text>
          </View>

          <Text style={styles.title}>ELIE</Text>
          <Text style={styles.subtitle}>Your Private AI</Text>

          {authMode === 'create' && (
            <TextInput
              style={styles.input}
              placeholder="Your name"
              placeholderTextColor="#777"
              value={name}
              onChangeText={setName}
            />
          )}

          <TextInput
            style={styles.input}
            placeholder="Username"
            placeholderTextColor="#777"
            value={username}
            onChangeText={setUsername}
            autoCapitalize="none"
          />

          <TextInput
            style={styles.input}
            placeholder="Password"
            placeholderTextColor="#777"
            value={password}
            onChangeText={setPassword}
            secureTextEntry
          />

          <Pressable
            style={styles.primaryButton}
            onPress={authMode === 'login' ? login : createProfile}>
            <Text style={styles.buttonText}>
              {authMode === 'login' ? 'Login' : 'Create Profile'}
            </Text>
          </Pressable>

          <Pressable
            onPress={() =>
              setAuthMode(
                authMode === 'login' ? 'create' : 'login',
              )
            }>
            <Text style={styles.switchText}>
              {authMode === 'login'
                ? "Don't have a profile? Create one"
                : 'Already have a profile? Login'}
            </Text>
          </Pressable>

          <Text style={styles.offline}>
            100% Offline • 100% Private
          </Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      {/* HEADER */}
      <View style={styles.header}>
        <Pressable
          style={styles.headerButton}
          onPress={() => setDrawerOpen(true)}>
          <Text style={styles.headerIcon}>☰</Text>
        </Pressable>

        <Text style={styles.headerTitle} numberOfLines={1}>
          {screen === 'chat'
            ? activeChat?.title || 'Elie'
            : screen === 'documents'
            ? 'Documents'
            : screen === 'settings'
            ? 'Settings'
            : 'Account'}
        </Text>

        <Pressable
          style={styles.headerButton}
          onPress={() => {
            if (screen === 'chat') createNewChat();
          }}>
          <Text style={styles.headerIcon}>＋</Text>
        </Pressable>
      </View>

      {/* CHAT */}
      {screen === 'chat' && (
        <View style={styles.chatContainer}>
          {!llm.isReady && !llm.error && (
            <View style={styles.loadingBar}>
              <Text style={styles.loadingText}>
                Loading Llama 3.2 1B...
              </Text>
              <Text style={styles.loadingPercent}>
                {Math.round(llm.downloadProgress * 100)}%
              </Text>
            </View>
          )}

          {llm.isReady && (
            <View style={styles.readyBar}>
              <Text style={styles.readyText}>
                ● Llama 3.2 1B • On-device
              </Text>
            </View>
          )}

          {!embeddings.isReady && !embeddings.error && (
            <View style={styles.embeddingBar}>
              <Text style={styles.embeddingText}>
                Loading document search model...
              </Text>
              <Text style={styles.loadingPercent}>
                {Math.round(embeddings.downloadProgress * 100)}%
              </Text>
            </View>
          )}

          {ragError ? (
            <View style={styles.errorBar}>
              <Text style={styles.errorText}>
                RAG: {ragError}
              </Text>
            </View>
          ) : null}

          {attachedDocuments.length > 0 && (
            <View style={styles.attachmentRow}>
              {attachedDocuments.map(document => (
                <View
                  key={document.id}
                  style={styles.documentChip}>
                  <Text
                    style={styles.documentChipText}
                    numberOfLines={1}>
                    📄 {document.name}
                  </Text>
                </View>
              ))}
            </View>
          )}

          {llm.isGenerating && (
            <View style={styles.generatingBar}>
              <Text style={styles.generatingText}>
                Elie is thinking...
              </Text>
            </View>
          )}

          {llm.error && (
            <View style={styles.errorBar}>
              <Text style={styles.errorText}>
                AI Error: {String(llm.error)}
              </Text>
            </View>
          )}

          <FlatList
            data={activeChat?.messages ?? []}
            keyExtractor={item => item.id}
            showsVerticalScrollIndicator={false}
            contentContainerStyle={styles.messages}
            renderItem={({item}) => (
              <View
                style={[
                  styles.messageBubble,
                  item.sender === 'user'
                    ? styles.userBubble
                    : styles.elieBubble,
                ]}>
                <View style={styles.messageHeader}>
                  <Text style={styles.sender}>
                    {item.sender === 'user' ? 'You' : 'Elie'}
                  </Text>
                  <Text style={styles.messageTime}>
                    {formatTime(item.createdAt)}
                  </Text>
                </View>

                <Text style={styles.messageText}>
                  {item.text}
                </Text>

                {item.sender === 'elie' && (
                  <Text style={styles.aiLabel}>
                    ● On-device AI
                  </Text>
                )}
              </View>
            )}
          />

          <View style={styles.inputRow}>
            <Pressable
              style={styles.attachButton}
              onPress={() => setAttachmentMenuVisible(true)}>
              <Text style={styles.attachText}>＋</Text>
            </Pressable>

            <TextInput
              style={styles.chatInput}
              placeholder={
                llm.isGenerating
                  ? 'Elie is thinking...'
                  : 'Ask Elie anything...'
              }
              placeholderTextColor="#666"
              value={message}
              onChangeText={setMessage}
              multiline
              editable={llm.isReady && !llm.isGenerating}
            />

            <Pressable
              style={[
                styles.sendButton,
                (!llm.isReady || llm.isGenerating) &&
                  styles.sendDisabled,
              ]}
              onPress={sendMessage}
              disabled={!llm.isReady || llm.isGenerating}>
              <Text style={styles.sendText}>
                {llm.isGenerating ? '...' : '➤'}
              </Text>
            </Pressable>
          </View>
        </View>
      )}

      {/* DOCUMENTS */}
      {screen === 'documents' && (
        <View style={styles.page}>
          <Text style={styles.pageTitle}>Documents</Text>

          <Pressable
            style={styles.primaryButton}
            onPress={addDocument}>
            <Text style={styles.buttonText}>
              + Add PDF / DOCX File
            </Text>
          </Pressable>

          {documents.length === 0 ? (
            <View style={styles.emptyCard}>
              <Text style={styles.emptyIcon}>📄</Text>
              <Text style={styles.emptyTitle}>No documents yet</Text>
              <Text style={styles.emptyText}>
                Add a PDF or DOCX file. Elie will extract, index and search it locally.
              </Text>
            </View>
          ) : (
            <FlatList
              data={documents}
              keyExtractor={item => item.id}
              showsVerticalScrollIndicator={false}
              style={{marginTop: 18}}
              renderItem={({item}) => (
                <View style={styles.documentCard}>
                  <Text style={styles.documentIcon}>📄</Text>

                  <View style={styles.documentInfo}>
                    <Text
                      style={styles.documentName}
                      numberOfLines={1}>
                      {item.name}
                    </Text>

                    <Text style={styles.documentSub}>
                      {item.status === 'ready'
                        ? item.indexStatus === 'indexed'
                          ? '✓ Ready for RAG'
                          : item.indexStatus === 'indexing'
                          ? 'Indexing...'
                          : 'Text extracted'
                        : item.status === 'empty'
                        ? 'No readable text'
                        : item.status === 'error'
                        ? 'Processing failed'
                        : 'Processing...'}
                    </Text>

                    {item.status === 'ready' && (
                      <View style={styles.documentActions}>
                        <Pressable
                          style={styles.viewTextButton}
                          onPress={() => viewDocumentText(item)}>
                          <Text style={styles.viewTextButtonText}>
                            View Text
                          </Text>
                        </Pressable>

                        {!attachedDocuments.some(
                          doc => doc.id === item.id,
                        ) && (
                          <Pressable
                            style={styles.attachSmallButton}
                            onPress={() => attachDocument(item.id)}>
                            <Text style={styles.attachSmallText}>
                              Attach
                            </Text>
                          </Pressable>
                        )}
                      </View>
                    )}
                  </View>

                  <Pressable onPress={() => removeDocument(item)}>
                    <Text style={styles.deleteIcon}>×</Text>
                  </Pressable>
                </View>
              )}
            />
          )}
        </View>
      )}

      {/* SETTINGS */}
      {screen === 'settings' && (
        <View style={styles.page}>
          <Text style={styles.pageTitle}>Settings</Text>

          <View style={styles.settingRow}>
            <View style={styles.settingInfo}>
              <Text style={styles.settingTitle}>Dark Mode</Text>
              <Text style={styles.settingSub}>
                Use Elie's dark interface
              </Text>
            </View>
            <Switch value={darkMode} onValueChange={setDarkMode} />
          </View>

          <View style={styles.settingRow}>
            <View style={styles.settingInfo}>
              <Text style={styles.settingTitle}>Offline AI</Text>
              <Text style={styles.settingSub}>
                Llama runs on this device
              </Text>
            </View>
            <Text style={styles.activeText}>ON</Text>
          </View>

          <View style={styles.settingRow}>
            <View style={styles.settingInfo}>
              <Text style={styles.settingTitle}>Local RAG</Text>
              <Text style={styles.settingSub}>
                Embeddings and vector search stay on this phone
              </Text>
            </View>
            <Text style={styles.activeText}>
              {ragReady ? 'ON' : 'OFF'}
            </Text>
          </View>

          <View style={styles.settingRow}>
            <View style={styles.settingInfo}>
              <Text style={styles.settingTitle}>Local Storage</Text>
              <Text style={styles.settingSub}>
                Your chats and documents stay on this phone
              </Text>
            </View>
            <Text style={styles.activeText}>ON</Text>
          </View>
        </View>
      )}

      {/* ACCOUNT */}
      {screen === 'account' && (
        <View style={styles.page}>
          <View style={styles.accountAvatar}>
            <Text style={styles.accountAvatarText}>
              {profile.name.charAt(0).toUpperCase()}
            </Text>
          </View>

          <Text style={styles.accountName}>{profile.name}</Text>
          <Text style={styles.accountUsername}>
            @{profile.username}
          </Text>
          <Text style={styles.localAccount}>
            Local Profile • Stored on device
          </Text>

          <Pressable style={styles.logoutButton} onPress={logout}>
            <Text style={styles.buttonText}>Logout</Text>
          </Pressable>
        </View>
      )}

      {/* SIDEBAR */}
      <Modal
        visible={drawerOpen}
        transparent
        animationType="slide"
        onRequestClose={() => setDrawerOpen(false)}>
        <View style={styles.drawerOverlay}>
          <View style={styles.drawer}>
            <View style={styles.drawerProfile}>
              <View style={styles.smallAvatar}>
                <Text style={styles.smallAvatarText}>
                  {profile.name.charAt(0).toUpperCase()}
                </Text>
              </View>

              <View style={styles.profileInfo}>
                <Text style={styles.drawerName}>{profile.name}</Text>
                <Text style={styles.drawerSub}>Local Profile</Text>
              </View>

              <Pressable
                style={styles.closeButton}
                onPress={() => setDrawerOpen(false)}>
                <Text style={styles.closeText}>×</Text>
              </Pressable>
            </View>

            <Pressable style={styles.newChatButton} onPress={createNewChat}>
              <Text style={styles.newChatText}>＋ New Chat</Text>
            </Pressable>

            <Text style={styles.sectionTitle}>Chat History</Text>

            <FlatList
              data={chats}
              keyExtractor={item => item.id}
              showsVerticalScrollIndicator={false}
              renderItem={({item}) => (
                <View
                  style={[
                    styles.historyRow,
                    item.id === activeChatId && styles.activeHistory,
                  ]}>
                  <Pressable
                    style={styles.historySelect}
                    onPress={() => {
                      setActiveChatId(item.id);
                      setScreen('chat');
                      setDrawerOpen(false);
                    }}>
                    <Text style={styles.historyIcon}>◉</Text>
                    <View style={{flex: 1}}>
                      <Text style={styles.historyText} numberOfLines={1}>
                        {item.title}
                      </Text>
                      <Text style={styles.historyTime}>
                        {formatTime(item.updatedAt)}
                      </Text>
                    </View>
                  </Pressable>

                  <Pressable
                    style={styles.chatMenuButton}
                    onPress={() => openChatMenu(item)}>
                    <Text style={styles.chatMenuDots}>⋮</Text>
                  </Pressable>
                </View>
              )}
            />

            <View style={styles.drawerBottom}>
              <Pressable
                style={styles.menuItem}
                onPress={() => {
                  setScreen('documents');
                  setDrawerOpen(false);
                }}>
                <Text style={styles.menuText}>📄 Documents</Text>
              </Pressable>

              <Pressable
                style={styles.menuItem}
                onPress={() => {
                  setScreen('settings');
                  setDrawerOpen(false);
                }}>
                <Text style={styles.menuText}>⚙ Settings</Text>
              </Pressable>

              <Pressable
                style={styles.menuItem}
                onPress={() => {
                  setScreen('account');
                  setDrawerOpen(false);
                }}>
                <Text style={styles.menuText}>👤 Account</Text>
              </Pressable>

              <Pressable style={styles.menuItem} onPress={logout}>
                <Text style={styles.logoutText}>⇥ Logout</Text>
              </Pressable>
            </View>
          </View>

          <Pressable
            style={styles.drawerOutside}
            onPress={() => setDrawerOpen(false)}
          />
        </View>
      </Modal>

      {/* ATTACHMENT MENU */}
      <Modal
        visible={attachmentMenuVisible}
        transparent
        animationType="fade"
        onRequestClose={() => setAttachmentMenuVisible(false)}>
        <Pressable
          style={styles.attachmentOverlay}
          onPress={() => setAttachmentMenuVisible(false)}>
          <View style={styles.attachmentMenu}>
            <Text style={styles.attachmentTitle}>Add to Elie</Text>

            <Pressable
              style={styles.attachmentOption}
              onPress={addDocument}>
              <Text style={styles.attachmentIcon}>📄</Text>
              <View style={{flex: 1}}>
                <Text style={styles.attachmentOptionTitle}>
                  Add Document
                </Text>
                <Text style={styles.attachmentOptionSub}>
                  PDF or DOCX • extract + index + RAG
                </Text>
              </View>
            </Pressable>

            {documents.length > 0 && (
              <View style={styles.existingDocuments}>
                <Text style={styles.existingTitle}>
                  Attach Existing Document
                </Text>

                {documents.map(document => (
                  <Pressable
                    key={document.id}
                    style={styles.existingDocument}
                    onPress={() => attachDocument(document.id)}>
                    <Text style={styles.attachmentIcon}>📄</Text>
                    <View style={{flex: 1}}>
                      <Text
                        style={styles.existingDocumentText}
                        numberOfLines={1}>
                        {document.name}
                      </Text>
                      <Text style={styles.existingDocumentStatus}>
                        {document.indexStatus === 'indexed'
                          ? 'Ready for RAG'
                          : document.status === 'ready'
                          ? 'Text extracted'
                          : 'Not ready'}
                      </Text>
                    </View>
                  </Pressable>
                ))}
              </View>
            )}

            <Pressable
              style={styles.attachmentCancel}
              onPress={() => setAttachmentMenuVisible(false)}>
              <Text style={styles.attachmentCancelText}>Cancel</Text>
            </Pressable>
          </View>
        </Pressable>
      </Modal>

      {/* CHAT ACTION MENU */}
      <Modal
        visible={chatMenuVisible}
        transparent
        animationType="fade"
        onRequestClose={() => setChatMenuVisible(false)}>
        <Pressable
          style={styles.actionOverlay}
          onPress={() => setChatMenuVisible(false)}>
          <View style={styles.actionCard}>
            <Text style={styles.actionTitle}>
              {selectedChat?.title}
            </Text>

            <Pressable style={styles.actionItem} onPress={startRename}>
              <Text style={styles.actionItemText}>✏ Rename Chat</Text>
            </Pressable>

            <Pressable
              style={styles.actionItem}
              onPress={deleteSelectedChat}>
              <Text style={styles.deleteActionText}>🗑 Delete Chat</Text>
            </Pressable>

            <Pressable
              style={styles.actionCancel}
              onPress={() => setChatMenuVisible(false)}>
              <Text style={styles.actionCancelText}>Cancel</Text>
            </Pressable>
          </View>
        </Pressable>
      </Modal>

      {/* RENAME */}
      <Modal
        visible={renameVisible}
        transparent
        animationType="fade"
        onRequestClose={() => setRenameVisible(false)}>
        <View style={styles.modalBackground}>
          <View style={styles.renameCard}>
            <Text style={styles.renameTitle}>Rename Chat</Text>

            <TextInput
              style={styles.input}
              value={renameValue}
              onChangeText={setRenameValue}
              placeholder="Chat name"
              placeholderTextColor="#777"
              autoFocus
            />

            <View style={styles.renameButtons}>
              <Pressable
                style={styles.cancelButton}
                onPress={() => setRenameVisible(false)}>
                <Text style={styles.cancelText}>Cancel</Text>
              </Pressable>

              <Pressable style={styles.saveButton} onPress={saveRename}>
                <Text style={styles.buttonText}>Save</Text>
              </Pressable>
            </View>
          </View>
        </View>
      </Modal>

      {/* DOCUMENT TEXT VIEWER */}
      <Modal
        visible={documentViewerVisible}
        animationType="slide"
        onRequestClose={() => setDocumentViewerVisible(false)}>
        <SafeAreaView style={styles.viewerContainer}>
          <View style={styles.viewerHeader}>
            <Text style={styles.viewerTitle} numberOfLines={1}>
              {documentViewerTitle}
            </Text>

            <Pressable onPress={() => setDocumentViewerVisible(false)}>
              <Text style={styles.viewerClose}>×</Text>
            </Pressable>
          </View>

          <FlatList
            data={[documentViewerText]}
            keyExtractor={() => 'document-text'}
            contentContainerStyle={styles.viewerContent}
            renderItem={({item}) => (
              <Text style={styles.viewerText}>{item}</Text>
            )}
          />
        </SafeAreaView>
      </Modal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {flex: 1, backgroundColor: '#08080B'},
  authContainer: {flex: 1, justifyContent: 'center', paddingHorizontal: 28},
  logo: {
    width: 82,
    height: 82,
    borderRadius: 23,
    backgroundColor: '#702BE0',
    alignSelf: 'center',
    justifyContent: 'center',
    alignItems: 'center',
  },
  logoText: {color: '#FFF', fontSize: 35, fontWeight: '800'},
  title: {
    color: '#FFF',
    fontSize: 36,
    fontWeight: '800',
    textAlign: 'center',
    letterSpacing: 5,
    marginTop: 16,
  },
  subtitle: {
    color: '#A78BFA',
    textAlign: 'center',
    fontSize: 16,
    marginBottom: 38,
  },
  input: {
    backgroundColor: '#15151B',
    borderWidth: 1,
    borderColor: '#292932',
    color: '#FFF',
    borderRadius: 13,
    padding: 15,
    marginBottom: 13,
    fontSize: 15,
  },
  primaryButton: {
    backgroundColor: '#702BE0',
    padding: 16,
    borderRadius: 13,
    alignItems: 'center',
    marginTop: 7,
  },
  buttonText: {color: '#FFF', fontSize: 15, fontWeight: '700'},
  switchText: {
    color: '#A78BFA',
    textAlign: 'center',
    marginTop: 20,
    fontSize: 14,
  },
  offline: {color: '#777', textAlign: 'center', marginTop: 25, fontSize: 13},
  header: {
    height: 62,
    borderBottomWidth: 1,
    borderBottomColor: '#202027',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 12,
  },
  headerButton: {
    width: 45,
    height: 45,
    alignItems: 'center',
    justifyContent: 'center',
  },
  headerIcon: {color: '#FFF', fontSize: 25},
  headerTitle: {color: '#FFF', fontSize: 17, fontWeight: '700', maxWidth: '70%'},
  chatContainer: {flex: 1, padding: 15},
  messages: {paddingBottom: 15},
  messageBubble: {
    padding: 15,
    borderRadius: 16,
    marginBottom: 12,
    maxWidth: '88%',
  },
  userBubble: {backgroundColor: '#5520A8', alignSelf: 'flex-end'},
  elieBubble: {backgroundColor: '#15151B', alignSelf: 'flex-start'},
  messageHeader: {flexDirection: 'row', alignItems: 'center', marginBottom: 6},
  sender: {color: '#C1A8FF', fontWeight: '700', flex: 1},
  messageTime: {color: '#777', fontSize: 10},
  messageText: {color: '#F4F4F4', fontSize: 15, lineHeight: 22},
  aiLabel: {color: '#8B7AAB', fontSize: 11, marginTop: 9},
  loadingBar: {
    backgroundColor: '#15151B',
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 9,
    marginBottom: 9,
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  loadingText: {color: '#A78BFA', fontSize: 12},
  loadingPercent: {color: '#FFF', fontSize: 12, fontWeight: '700'},
  readyBar: {
    backgroundColor: '#15151B',
    borderRadius: 10,
    paddingVertical: 8,
    paddingHorizontal: 12,
    marginBottom: 9,
  },
  readyText: {color: '#A78BFA', fontSize: 11},
  embeddingBar: {
    backgroundColor: '#121A20',
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 9,
    marginBottom: 9,
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  embeddingText: {color: '#9FD8FF', fontSize: 11},
  generatingBar: {
    backgroundColor: '#191421',
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 8,
    marginBottom: 9,
  },
  generatingText: {color: '#B99AFF', fontSize: 11},
  errorBar: {
    backgroundColor: '#2A1518',
    borderRadius: 10,
    padding: 10,
    marginBottom: 9,
  },
  errorText: {color: '#E77777', fontSize: 11},
  attachmentRow: {flexDirection: 'row', flexWrap: 'wrap', marginBottom: 8},
  documentChip: {
    backgroundColor: '#1C1628',
    borderWidth: 1,
    borderColor: '#35244D',
    paddingHorizontal: 10,
    paddingVertical: 7,
    borderRadius: 12,
    marginRight: 6,
    marginBottom: 5,
    maxWidth: '80%',
  },
  documentChipText: {color: '#C4B5FD', fontSize: 11},
  inputRow: {flexDirection: 'row', alignItems: 'flex-end'},
  attachButton: {
    width: 48,
    height: 48,
    borderRadius: 14,
    backgroundColor: '#15151B',
    borderWidth: 1,
    borderColor: '#292932',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 8,
  },
  attachText: {color: '#C4B5FD', fontSize: 26, lineHeight: 28},
  chatInput: {
    flex: 1,
    maxHeight: 100,
    backgroundColor: '#15151B',
    borderRadius: 15,
    paddingHorizontal: 14,
    paddingVertical: 13,
    color: '#FFF',
    borderWidth: 1,
    borderColor: '#292932',
  },
  sendButton: {
    width: 51,
    height: 51,
    borderRadius: 14,
    backgroundColor: '#702BE0',
    marginLeft: 8,
    alignItems: 'center',
    justifyContent: 'center',
  },
  sendDisabled: {opacity: 0.45},
  sendText: {color: '#FFF', fontSize: 22},
  page: {flex: 1, padding: 20},
  pageTitle: {color: '#FFF', fontSize: 28, fontWeight: '800', marginBottom: 20},
  emptyCard: {
    backgroundColor: '#15151B',
    borderRadius: 16,
    padding: 25,
    alignItems: 'center',
    marginTop: 20,
  },
  emptyIcon: {fontSize: 40},
  emptyTitle: {color: '#FFF', fontSize: 18, fontWeight: '700', marginTop: 10},
  emptyText: {color: '#777', textAlign: 'center', marginTop: 8, lineHeight: 19},
  documentCard: {
    backgroundColor: '#15151B',
    borderRadius: 14,
    padding: 15,
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  documentIcon: {fontSize: 26, marginRight: 12},
  documentInfo: {flex: 1},
  documentName: {color: '#FFF', fontSize: 15, fontWeight: '700'},
  documentSub: {color: '#777', fontSize: 12, marginTop: 4},
  documentActions: {flexDirection: 'row', marginTop: 7},
  viewTextButton: {
    alignSelf: 'flex-start',
    paddingHorizontal: 9,
    paddingVertical: 5,
    backgroundColor: '#281A3B',
    borderRadius: 8,
  },
  viewTextButtonText: {color: '#C4B5FD', fontSize: 11, fontWeight: '700'},
  attachSmallButton: {
    alignSelf: 'flex-start',
    paddingHorizontal: 9,
    paddingVertical: 5,
    backgroundColor: '#1A2330',
    borderRadius: 8,
    marginLeft: 6,
  },
  attachSmallText: {color: '#9FD8FF', fontSize: 11, fontWeight: '700'},
  deleteIcon: {color: '#E77777', fontSize: 27, paddingHorizontal: 8},
  settingRow: {
    backgroundColor: '#15151B',
    borderRadius: 14,
    padding: 17,
    marginBottom: 10,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  settingInfo: {flex: 1},
  settingTitle: {color: '#FFF', fontSize: 15, fontWeight: '600'},
  settingSub: {color: '#777', fontSize: 12, marginTop: 4},
  activeText: {color: '#A78BFA', fontWeight: '800'},
  accountAvatar: {
    width: 100,
    height: 100,
    borderRadius: 50,
    backgroundColor: '#702BE0',
    alignSelf: 'center',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 30,
  },
  accountAvatarText: {color: '#FFF', fontSize: 40, fontWeight: '800'},
  accountName: {
    color: '#FFF',
    fontSize: 25,
    fontWeight: '700',
    textAlign: 'center',
    marginTop: 18,
  },
  accountUsername: {color: '#A78BFA', textAlign: 'center', marginTop: 5},
  localAccount: {color: '#777', textAlign: 'center', marginTop: 8},
  logoutButton: {
    backgroundColor: '#702BE0',
    padding: 16,
    borderRadius: 13,
    alignItems: 'center',
    marginTop: 30,
  },
  drawerOverlay: {
    flex: 1,
    flexDirection: 'row',
    backgroundColor: 'rgba(0,0,0,0.68)',
  },
  drawer: {
    width: '88%',
    height: '100%',
    backgroundColor: '#0D0D12',
    paddingTop: 24,
    paddingHorizontal: 18,
    paddingBottom: 12,
    borderTopRightRadius: 20,
    borderBottomRightRadius: 20,
    elevation: 20,
  },
  drawerOutside: {flex: 1},
  drawerProfile: {flexDirection: 'row', alignItems: 'center', marginBottom: 22},
  smallAvatar: {
    width: 48,
    height: 48,
    borderRadius: 15,
    backgroundColor: '#702BE0',
    alignItems: 'center',
    justifyContent: 'center',
  },
  smallAvatarText: {color: '#FFF', fontSize: 21, fontWeight: '800'},
  profileInfo: {flex: 1, marginLeft: 12},
  drawerName: {color: '#FFF', fontSize: 17, fontWeight: '700'},
  drawerSub: {color: '#777', fontSize: 12, marginTop: 3},
  closeButton: {
    width: 38,
    height: 38,
    borderRadius: 12,
    backgroundColor: '#15151B',
    alignItems: 'center',
    justifyContent: 'center',
  },
  closeText: {color: '#AAA', fontSize: 28},
  newChatButton: {
    backgroundColor: '#702BE0',
    padding: 15,
    borderRadius: 12,
    marginBottom: 20,
  },
  newChatText: {color: '#FFF', fontSize: 15, fontWeight: '700'},
  sectionTitle: {
    color: '#777',
    fontSize: 11,
    textTransform: 'uppercase',
    letterSpacing: 1.2,
    marginBottom: 8,
  },
  historyRow: {flexDirection: 'row', alignItems: 'center', borderRadius: 10, marginBottom: 4},
  activeHistory: {backgroundColor: '#1B1525'},
  historySelect: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
    paddingVertical: 11,
    paddingHorizontal: 10,
  },
  historyIcon: {color: '#A78BFA', marginRight: 10, fontSize: 12},
  historyText: {color: '#DDD', fontSize: 14},
  historyTime: {color: '#666', fontSize: 10, marginTop: 3},
  chatMenuButton: {
    width: 40,
    height: 48,
    alignItems: 'center',
    justifyContent: 'center',
  },
  chatMenuDots: {color: '#AAA', fontSize: 22},
  drawerBottom: {borderTopWidth: 1, borderTopColor: '#24242B', paddingTop: 8, marginTop: 'auto'},
  menuItem: {paddingVertical: 13, paddingHorizontal: 4},
  menuText: {color: '#DDD', fontSize: 15},
  logoutText: {color: '#D77777', fontSize: 15},
  attachmentOverlay: {flex: 1, backgroundColor: 'rgba(0,0,0,0.65)', justifyContent: 'flex-end', padding: 15},
  attachmentMenu: {backgroundColor: '#15151B', borderRadius: 20, padding: 18, borderWidth: 1, borderColor: '#292932'},
  attachmentTitle: {color: '#FFF', fontSize: 18, fontWeight: '700', marginBottom: 15},
  attachmentOption: {flexDirection: 'row', alignItems: 'center', padding: 14, borderRadius: 13, backgroundColor: '#1C1C24'},
  attachmentIcon: {fontSize: 25, marginRight: 13},
  attachmentOptionTitle: {color: '#FFF', fontSize: 15, fontWeight: '700'},
  attachmentOptionSub: {color: '#777', fontSize: 12, marginTop: 3},
  existingDocuments: {marginTop: 16},
  existingTitle: {color: '#777', fontSize: 11, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8},
  existingDocument: {flexDirection: 'row', alignItems: 'center', backgroundColor: '#1C1C24', borderRadius: 11, padding: 12, marginBottom: 7},
  existingDocumentText: {color: '#DDD', fontSize: 13},
  existingDocumentStatus: {color: '#777', fontSize: 10, marginTop: 2},
  attachmentCancel: {padding: 14, marginTop: 10},
  attachmentCancelText: {color: '#999', textAlign: 'center', fontWeight: '600'},
  actionOverlay: {flex: 1, backgroundColor: 'rgba(0,0,0,0.65)', justifyContent: 'center', paddingHorizontal: 30},
  actionCard: {backgroundColor: '#15151B', borderRadius: 18, padding: 18, borderWidth: 1, borderColor: '#292932'},
  actionTitle: {color: '#FFF', fontSize: 17, fontWeight: '700', marginBottom: 12},
  actionItem: {paddingVertical: 15, borderBottomWidth: 1, borderBottomColor: '#24242B'},
  actionItemText: {color: '#FFF', fontSize: 15},
  deleteActionText: {color: '#E77777', fontSize: 15},
  actionCancel: {paddingVertical: 15},
  actionCancelText: {color: '#999', textAlign: 'center', fontWeight: '600'},
  modalBackground: {flex: 1, backgroundColor: 'rgba(0,0,0,0.72)', justifyContent: 'center', padding: 25},
  renameCard: {backgroundColor: '#121218', borderRadius: 18, padding: 20, borderWidth: 1, borderColor: '#292932'},
  renameTitle: {color: '#FFF', fontSize: 21, fontWeight: '800', marginBottom: 18},
  renameButtons: {flexDirection: 'row', justifyContent: 'flex-end', alignItems: 'center'},
  cancelButton: {paddingVertical: 11, paddingHorizontal: 15, marginRight: 8},
  cancelText: {color: '#999', fontWeight: '600'},
  saveButton: {backgroundColor: '#702BE0', paddingVertical: 11, paddingHorizontal: 20, borderRadius: 10},
  viewerContainer: {flex: 1, backgroundColor: '#08080B'},
  viewerHeader: {height: 62, paddingHorizontal: 15, borderBottomWidth: 1, borderBottomColor: '#202027', flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between'},
  viewerTitle: {color: '#FFF', fontSize: 17, fontWeight: '700', flex: 1, marginRight: 15},
  viewerClose: {color: '#AAA', fontSize: 29},
  viewerContent: {padding: 18},
  viewerText: {color: '#E8E8E8', fontSize: 14, lineHeight: 22},
});
