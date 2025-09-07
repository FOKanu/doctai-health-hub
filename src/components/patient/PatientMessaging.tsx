import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import {
  MessageSquare,
  Send,
  Search,
  Plus,
  Clock,
  User,
  AlertCircle,
  CheckCircle,
  Star
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { useAuth } from '../../contexts/AuthContext';
import {
  getMessageThreads,
  getThreadMessages,
  sendMessage,
  markMessageAsRead,
  MessageThread,
  Message
} from '../../services/messagingService';

export function PatientMessaging() {
  const { user } = useAuth();
  const { toast } = useToast();
  const [threads, setThreads] = useState<MessageThread[]>([]);
  const [selectedThread, setSelectedThread] = useState<MessageThread | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isSending, setIsSending] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [newMessage, setNewMessage] = useState('');
  const [showCompose, setShowCompose] = useState(false);
  const [composeForm, setComposeForm] = useState({
    recipient: '',
    subject: '',
    content: ''
  });

  useEffect(() => {
    loadMessageThreads();
  }, []);

  useEffect(() => {
    if (selectedThread) {
      loadThreadMessages(selectedThread.id);
    }
  }, [selectedThread]);

  const loadMessageThreads = async () => {
    try {
      setIsLoading(true);
      const data = await getMessageThreads(user?.id || '');
      setThreads(data);
    } catch (error) {
      console.error('Error loading message threads:', error);
      toast({
        title: "Error",
        description: "Failed to load messages.",
        variant: "destructive"
      });
    } finally {
      setIsLoading(false);
    }
  };

  const loadThreadMessages = async (threadId: string) => {
    try {
      const data = await getThreadMessages(threadId);
      setMessages(data);

      // Mark messages as read
      data.forEach(message => {
        if (message.recipientId === user?.id && message.status !== 'read') {
          markMessageAsRead(message.id);
        }
      });
    } catch (error) {
      console.error('Error loading thread messages:', error);
    }
  };

  const handleSendMessage = async () => {
    if (!selectedThread || !newMessage.trim()) return;

    try {
      setIsSending(true);

      const message: Omit<Message, 'id' | 'timestamp' | 'status'> = {
        threadId: selectedThread.id,
        senderId: user?.id || '',
        senderName: user?.name || 'Patient',
        senderType: 'patient',
        recipientId: selectedThread.participants.find(p => p.id !== user?.id)?.id || '',
        recipientName: selectedThread.participants.find(p => p.id !== user?.id)?.name || '',
        subject: selectedThread.subject,
        content: newMessage.trim(),
        priority: 'normal',
        messageType: 'text'
      };

      const success = await sendMessage(message);

      if (success) {
        setNewMessage('');
        await loadThreadMessages(selectedThread.id);
        await loadMessageThreads(); // Refresh threads to update last message
        toast({
          title: "Message Sent",
          description: "Your message has been sent successfully.",
        });
      } else {
        throw new Error('Failed to send message');
      }
    } catch (error) {
      console.error('Error sending message:', error);
      toast({
        title: "Error",
        description: "Failed to send message. Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsSending(false);
    }
  };

  const handleComposeMessage = async () => {
    if (!composeForm.recipient || !composeForm.subject || !composeForm.content.trim()) {
      toast({
        title: "Missing Information",
        description: "Please fill in all required fields.",
        variant: "destructive"
      });
      return;
    }

    try {
      setIsSending(true);

      // For MVP, we'll create a simple message
      const message: Omit<Message, 'id' | 'timestamp' | 'status'> = {
        threadId: `thread_${Date.now()}`,
        senderId: user?.id || '',
        senderName: user?.name || 'Patient',
        senderType: 'patient',
        recipientId: 'provider_001', // Mock provider ID
        recipientName: composeForm.recipient,
        subject: composeForm.subject,
        content: composeForm.content.trim(),
        priority: 'normal',
        messageType: 'text'
      };

      const success = await sendMessage(message);

      if (success) {
        setComposeForm({ recipient: '', subject: '', content: '' });
        setShowCompose(false);
        await loadMessageThreads();
        toast({
          title: "Message Sent",
          description: "Your message has been sent successfully.",
        });
      } else {
        throw new Error('Failed to send message');
      }
    } catch (error) {
      console.error('Error sending message:', error);
      toast({
        title: "Error",
        description: "Failed to send message. Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsSending(false);
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'urgent': return 'bg-red-100 text-red-800';
      case 'emergency': return 'bg-red-200 text-red-900';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const hours = diff / (1000 * 60 * 60);

    if (hours < 1) return 'Just now';
    if (hours < 24) return `${Math.floor(hours)}h ago`;
    return date.toLocaleDateString();
  };

  const filteredThreads = threads.filter(thread =>
    thread.subject.toLowerCase().includes(searchTerm.toLowerCase()) ||
    thread.participants.some(p => p.name.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading messages...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Messages</h1>
          <p className="text-muted-foreground mt-1">Communicate with your healthcare providers</p>
        </div>
        <Button onClick={() => setShowCompose(true)} className="flex items-center space-x-2">
          <Plus className="w-4 h-4" />
          <span>New Message</span>
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Threads List */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <MessageSquare className="w-5 h-5" />
              <span>Conversations</span>
            </CardTitle>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
              <Input
                placeholder="Search conversations..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10"
              />
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {filteredThreads.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  <MessageSquare className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                  <p>No conversations found</p>
                </div>
              ) : (
                filteredThreads.map((thread) => (
                  <div
                    key={thread.id}
                    className={`p-3 border rounded-lg cursor-pointer transition-colors ${
                      selectedThread?.id === thread.id
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                    onClick={() => setSelectedThread(thread)}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <Avatar className="w-8 h-8">
                          <AvatarFallback className="text-xs">
                            {thread.participants.find(p => p.id !== user?.id)?.name.split(' ').map(n => n[0]).join('') || 'P'}
                          </AvatarFallback>
                        </Avatar>
                        <div>
                          <p className="font-medium text-sm">
                            {thread.participants.find(p => p.id !== user?.id)?.name || 'Unknown'}
                          </p>
                          <p className="text-xs text-gray-500">
                            {thread.participants.find(p => p.id !== user?.id)?.type || 'provider'}
                          </p>
                        </div>
                      </div>
                      {thread.unreadCount > 0 && (
                        <Badge variant="destructive" className="text-xs">
                          {thread.unreadCount}
                        </Badge>
                      )}
                    </div>

                    <p className="text-sm font-medium text-gray-900 mb-1">{thread.subject}</p>
                    <p className="text-xs text-gray-600 line-clamp-2">{thread.lastMessage.content}</p>

                    <div className="flex items-center justify-between mt-2">
                      <span className="text-xs text-gray-500">
                        {formatTimestamp(thread.lastMessage.timestamp)}
                      </span>
                      {thread.priority !== 'normal' && (
                        <Badge className={getPriorityColor(thread.priority)}>
                          {thread.priority}
                        </Badge>
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>
          </CardContent>
        </Card>

        {/* Messages */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>
              {selectedThread ? selectedThread.subject : 'Select a conversation'}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {selectedThread ? (
              <div className="space-y-4">
                {/* Messages List */}
                <div className="space-y-4 max-h-96 overflow-y-auto">
                  {messages.map((message) => (
                    <div
                      key={message.id}
                      className={`flex ${message.senderId === user?.id ? 'justify-end' : 'justify-start'}`}
                    >
                      <div className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                        message.senderId === user?.id
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-100 text-gray-900'
                      }`}>
                        <p className="text-sm">{message.content}</p>
                        <p className={`text-xs mt-1 ${
                          message.senderId === user?.id ? 'text-blue-100' : 'text-gray-500'
                        }`}>
                          {formatTimestamp(message.timestamp)}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Message Input */}
                <div className="flex space-x-2">
                  <Textarea
                    placeholder="Type your message..."
                    value={newMessage}
                    onChange={(e) => setNewMessage(e.target.value)}
                    className="flex-1"
                    rows={2}
                  />
                  <Button
                    onClick={handleSendMessage}
                    disabled={isSending || !newMessage.trim()}
                    className="self-end"
                  >
                    <Send className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            ) : (
              <div className="text-center py-12 text-gray-500">
                <MessageSquare className="w-16 h-16 mx-auto mb-4 text-gray-300" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No conversation selected</h3>
                <p className="text-gray-500">Select a conversation from the list to start messaging.</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Compose Modal */}
      {showCompose && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <Card className="w-full max-w-md mx-4">
            <CardHeader>
              <CardTitle>Compose Message</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm font-medium">To</label>
                <Input
                  placeholder="Provider name"
                  value={composeForm.recipient}
                  onChange={(e) => setComposeForm(prev => ({ ...prev, recipient: e.target.value }))}
                />
              </div>
              <div>
                <label className="text-sm font-medium">Subject</label>
                <Input
                  placeholder="Message subject"
                  value={composeForm.subject}
                  onChange={(e) => setComposeForm(prev => ({ ...prev, subject: e.target.value }))}
                />
              </div>
              <div>
                <label className="text-sm font-medium">Message</label>
                <Textarea
                  placeholder="Type your message..."
                  value={composeForm.content}
                  onChange={(e) => setComposeForm(prev => ({ ...prev, content: e.target.value }))}
                  rows={4}
                />
              </div>
              <div className="flex space-x-2">
                <Button
                  onClick={handleComposeMessage}
                  disabled={isSending}
                  className="flex-1"
                >
                  {isSending ? 'Sending...' : 'Send Message'}
                </Button>
                <Button
                  variant="outline"
                  onClick={() => setShowCompose(false)}
                  className="flex-1"
                >
                  Cancel
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
