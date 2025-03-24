import { useState } from 'react';
import { useImmer } from 'use-immer';
import api from '@/api';
import { parseSSEStream } from '@/utils';
import ChatMessages from '@/components/ChatMessages';
import ChatInput from '@/components/ChatInput';

function Chatbot() {
  const [chatId, setChatId] = useState(null);
  const [messages, setMessages] = useImmer([]);
  const [newMessage, setNewMessage] = useState('');

  const isLoading = messages.length && messages[messages.length - 1].loading;

async function submitNewMessage() {
  const trimmedMessage = newMessage.trim();
  if (!trimmedMessage || isLoading) return;

  setMessages(draft => {
    draft.push({ role: 'user', content: trimmedMessage });
    draft.push({ role: 'assistant', content: '', loading: true });
  });
  setNewMessage('');

  try {
    const reply = await api.sendChatMessage(trimmedMessage);

    setMessages(draft => {
      draft[draft.length - 1].content = reply;
      draft[draft.length - 1].loading = false;
    });
  } catch (err) {
    console.log(err);
    setMessages(draft => {
      draft[draft.length - 1].loading = false;
      draft[draft.length - 1].error = true;
    });
  }
}


  return (
    <div className='relative grow flex flex-col gap-6 pt-6'>
      {messages.length === 0 && (
        <div className='mt-3 font-urbanist text-primary-blue text-xl font-light space-y-2'>
          <p>👋 Welcome!</p>
          <p>I am a nutrition chatbot. I can help you with your nutrition questions. Just ask about anything related to nutrition!</p>
        </div>
      )}
      <ChatMessages
        messages={messages}
        isLoading={isLoading}
      />
      <ChatInput
        newMessage={newMessage}
        isLoading={isLoading}
        setNewMessage={setNewMessage}
        submitNewMessage={submitNewMessage}
      />
    </div>
  );
}

export default Chatbot;