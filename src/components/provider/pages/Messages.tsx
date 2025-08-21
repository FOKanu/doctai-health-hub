import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { 
  MessageSquare, 
  Send, 
  Search, 
  Plus,
  Clock,
  User,
  Users,
  Star,
  Archive,
  Trash2,
  Reply,
  Forward,
  PaperclipIcon,
  Filter
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface Message {
  id: string;
  subject: string;
  content: string;
  sender: {
    id: string;
    name: string;
    type: 'patient' | 'provider' | 'staff';
    initials: string;
  };
  recipient: {
    id: string;
    name: string;
    initials: string;
  };
  timestamp: Date;
  status: 'unread' | 'read' | 'replied';
  priority: 'normal' | 'urgent';
  thread: string;
  attachments?: number;
}

interface MessageThread {
  id: string;
  subject: string;
  participants: Array<{
    id: string;
    name: string;
    initials: string;
    type: 'patient' | 'provider' | 'staff';
  }>;
  lastMessage: Date;
  unreadCount: number;
  messages: Message[];
}

// Mock message data
const mockMessages: Message[] = [
  {
    id: 'M001',
    subject: 'Lab Results Follow-up',
    content: 'Dr. Smith, I received my lab results and have some questions about my cholesterol levels. Could we discuss the next steps?',
    sender: { id: 'P001', name: 'Sarah Johnson', type: 'patient', initials: 'SJ' },
    recipient: { id: 'DR001', name: 'Dr. Michael Chen', initials: 'MC' },
    timestamp: new Date(2024, 1, 20, 14, 30),
    status: 'unread',
    priority: 'normal',
    thread: 'T001'
  },
  {
    id: 'M002',
    subject: 'Medication Adjustment Request',
    content: 'Patient is experiencing side effects from current medication. Requesting consultation for adjustment.',
    sender: { id: 'N001', name: 'Nurse Williams', type: 'staff', initials: 'NW' },
    recipient: { id: 'DR001', name: 'Dr. Michael Chen', initials: 'MC' },
    timestamp: new Date(2024, 1, 20, 13, 15),
    status: 'read',
    priority: 'urgent',
    thread: 'T002'
  },
  {
    id: 'M003',
    subject: 'Appointment Confirmation',
    content: 'Confirming appointment scheduled for tomorrow at 2:00 PM. Please bring your insurance card and any new medications.',
    sender: { id: 'DR001', name: 'Dr. Michael Chen', type: 'provider', initials: 'MC' },
    recipient: { id: 'P002', name: 'Michael Chen', initials: 'MC' },
    timestamp: new Date(2024, 1, 20, 11, 45),
    status: 'read',
    priority: 'normal',
    thread: 'T003'
  },
  {
    id: 'M004',
    subject: 'Test Results - Urgent Review',
    content: 'Abnormal findings in recent CT scan. Please review immediately and advise on next steps.',
    sender: { id: 'RAD001', name: 'Dr. Radiologist', type: 'provider', initials: 'DR' },
    recipient: { id: 'DR001', name: 'Dr. Michael Chen', initials: 'MC' },
    timestamp: new Date(2024, 1, 20, 9, 20),
    status: 'unread',
    priority: 'urgent',
    thread: 'T004',
    attachments: 2
  }
];

export function Messages() {
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState('inbox');
  const [selectedMessage, setSelectedMessage] = useState<Message | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('all');
  const [isComposing, setIsComposing] = useState(false);
  const [replyContent, setReplyContent] = useState('');

  // Compose form state
  const [composeForm, setComposeForm] = useState({
    recipient: '',
    subject: '',
    content: '',
    priority: 'normal' as 'normal' | 'urgent'
  });

  const filteredMessages = mockMessages.filter(message => {
    const matchesSearch = message.subject.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         message.sender.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         message.content.toLowerCase().includes(searchTerm.toLowerCase());
    
    const matchesFilter = filterType === 'all' || 
                         (filterType === 'patients' && message.sender.type === 'patient') ||
                         (filterType === 'providers' && message.sender.type === 'provider') ||
                         (filterType === 'staff' && message.sender.type === 'staff') ||
                         (filterType === 'urgent' && message.priority === 'urgent') ||
                         (filterType === 'unread' && message.status === 'unread');

    return matchesSearch && matchesFilter;
  });

  const unreadCount = mockMessages.filter(m => m.status === 'unread').length;
  const urgentCount = mockMessages.filter(m => m.priority === 'urgent').length;

  const handleSendMessage = () => {
    if (!composeForm.recipient || !composeForm.subject || !composeForm.content) {
      toast({
        title: "Missing Information",
        description: "Please fill in all required fields.",
        variant: "destructive",
      });
      return;
    }

    toast({
      title: "Message Sent",
      description: `Message sent to ${composeForm.recipient}`,
    });

    setComposeForm({ recipient: '', subject: '', content: '', priority: 'normal' });
    setIsComposing(false);
  };

  const handleReply = () => {
    if (!replyContent.trim()) return;

    toast({
      title: "Reply Sent",
      description: "Your reply has been sent successfully.",
    });

    setReplyContent('');
  };

  const getSenderTypeColor = (type: string) => {
    switch (type) {
      case 'patient': return 'bg-blue-500';
      case 'provider': return 'bg-green-500';
      case 'staff': return 'bg-purple-500';
      default: return 'bg-gray-500';
    }
  };

  const formatTimestamp = (date: Date) => {
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const hours = diff / (1000 * 60 * 60);
    
    if (hours < 1) return 'Just now';
    if (hours < 24) return `${Math.floor(hours)}h ago`;
    return date.toLocaleDateString();
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Messages</h1>
          <p className="text-muted-foreground mt-1">Secure communication with patients and team</p>
        </div>
        
        <div className="flex items-center space-x-3">
          <div className="relative">
            <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground" />
            <Input 
              placeholder="Search messages..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-9 w-80"
            />
          </div>
          <Button onClick={() => setIsComposing(true)}>
            <Plus className="w-4 h-4 mr-2" />
            Compose
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Sidebar */}
        <div className="space-y-4">
          {/* Quick Stats */}
          <Card>
            <CardContent className="p-4">
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Unread</span>
                  <Badge variant="destructive">{unreadCount}</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Urgent</span>
                  <Badge variant="default">{urgentCount}</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Total</span>
                  <span className="font-medium">{mockMessages.length}</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Filters */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2 text-sm">
                <Filter className="w-4 h-4" />
                <span>Filter Messages</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="p-4 pt-0">
              <Select value={filterType} onValueChange={setFilterType}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Messages</SelectItem>
                  <SelectItem value="patients">From Patients</SelectItem>
                  <SelectItem value="providers">From Providers</SelectItem>
                  <SelectItem value="staff">From Staff</SelectItem>
                  <SelectItem value="urgent">Urgent Only</SelectItem>
                  <SelectItem value="unread">Unread Only</SelectItem>
                </SelectContent>
              </Select>
            </CardContent>
          </Card>

          {/* Navigation */}
          <Card>
            <CardContent className="p-4">
              <div className="space-y-2">
                <Button 
                  variant={activeTab === 'inbox' ? 'default' : 'ghost'} 
                  className="w-full justify-start"
                  onClick={() => setActiveTab('inbox')}
                >
                  <MessageSquare className="w-4 h-4 mr-2" />
                  Inbox
                </Button>
                <Button 
                  variant="ghost" 
                  className="w-full justify-start"
                >
                  <Send className="w-4 h-4 mr-2" />
                  Sent
                </Button>
                <Button 
                  variant="ghost" 
                  className="w-full justify-start"
                >
                  <Archive className="w-4 h-4 mr-2" />
                  Archive
                </Button>
                <Button 
                  variant="ghost" 
                  className="w-full justify-start"
                >
                  <Star className="w-4 h-4 mr-2" />
                  Starred
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Content */}
        <div className="lg:col-span-3">
          {isComposing ? (
            /* Compose Message */
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span>Compose Message</span>
                  <Button variant="outline" onClick={() => setIsComposing(false)}>
                    Cancel
                  </Button>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-sm font-medium">Recipient *</label>
                    <Select value={composeForm.recipient} onValueChange={(value) => 
                      setComposeForm(prev => ({ ...prev, recipient: value }))
                    }>
                      <SelectTrigger className="mt-1">
                        <SelectValue placeholder="Select recipient" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="sarah-johnson">Sarah Johnson (Patient)</SelectItem>
                        <SelectItem value="nurse-williams">Nurse Williams (Staff)</SelectItem>
                        <SelectItem value="dr-radiologist">Dr. Radiologist (Provider)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  
                  <div>
                    <label className="text-sm font-medium">Priority</label>
                    <Select value={composeForm.priority} onValueChange={(value: 'normal' | 'urgent') => 
                      setComposeForm(prev => ({ ...prev, priority: value }))
                    }>
                      <SelectTrigger className="mt-1">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="normal">Normal</SelectItem>
                        <SelectItem value="urgent">Urgent</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div>
                  <label className="text-sm font-medium">Subject *</label>
                  <Input
                    value={composeForm.subject}
                    onChange={(e) => setComposeForm(prev => ({ ...prev, subject: e.target.value }))}
                    className="mt-1"
                    placeholder="Enter subject..."
                  />
                </div>

                <div>
                  <label className="text-sm font-medium">Message *</label>
                  <Textarea
                    value={composeForm.content}
                    onChange={(e) => setComposeForm(prev => ({ ...prev, content: e.target.value }))}
                    className="mt-1"
                    rows={8}
                    placeholder="Type your message..."
                  />
                </div>

                <div className="flex items-center justify-between">
                  <Button variant="outline">
                    <PaperclipIcon className="w-4 h-4 mr-2" />
                    Attach File
                  </Button>
                  
                  <Button onClick={handleSendMessage}>
                    <Send className="w-4 h-4 mr-2" />
                    Send Message
                  </Button>
                </div>
              </CardContent>
            </Card>
          ) : selectedMessage ? (
            /* Message Detail View */
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <Button variant="ghost" onClick={() => setSelectedMessage(null)}>
                      ← Back
                    </Button>
                    <div>
                      <h2 className="font-semibold">{selectedMessage.subject}</h2>
                      {selectedMessage.priority === 'urgent' && (
                        <Badge variant="destructive" className="mt-1">Urgent</Badge>
                      )}
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <Button variant="outline" size="sm">
                      <Reply className="w-4 h-4 mr-1" />
                      Reply
                    </Button>
                    <Button variant="outline" size="sm">
                      <Forward className="w-4 h-4 mr-1" />
                      Forward
                    </Button>
                    <Button variant="outline" size="sm">
                      <Archive className="w-4 h-4 mr-1" />
                      Archive
                    </Button>
                  </div>
                </div>
              </CardHeader>
              
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center space-x-3 p-4 bg-muted/50 rounded-lg">
                    <Avatar>
                      <AvatarFallback className={getSenderTypeColor(selectedMessage.sender.type)}>
                        {selectedMessage.sender.initials}
                      </AvatarFallback>
                    </Avatar>
                    <div className="flex-1">
                      <div className="flex items-center space-x-2">
                        <span className="font-medium">{selectedMessage.sender.name}</span>
                        <Badge variant="outline" className="text-xs capitalize">
                          {selectedMessage.sender.type}
                        </Badge>
                      </div>
                      <div className="text-sm text-muted-foreground flex items-center space-x-2">
                        <Clock className="w-3 h-3" />
                        <span>{selectedMessage.timestamp.toLocaleString()}</span>
                        {selectedMessage.attachments && (
                          <>
                            <PaperclipIcon className="w-3 h-3" />
                            <span>{selectedMessage.attachments} attachment{selectedMessage.attachments > 1 ? 's' : ''}</span>
                          </>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="prose max-w-none">
                    <p className="whitespace-pre-wrap">{selectedMessage.content}</p>
                  </div>

                  {/* Reply Section */}
                  <div className="border-t pt-4">
                    <label className="text-sm font-medium">Reply</label>
                    <Textarea
                      value={replyContent}
                      onChange={(e) => setReplyContent(e.target.value)}
                      className="mt-2"
                      rows={4}
                      placeholder="Type your reply..."
                    />
                    <div className="flex items-center justify-end space-x-2 mt-3">
                      <Button variant="outline">
                        <PaperclipIcon className="w-4 h-4 mr-2" />
                        Attach
                      </Button>
                      <Button onClick={handleReply}>
                        <Reply className="w-4 h-4 mr-2" />
                        Send Reply
                      </Button>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          ) : (
            /* Message List */
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <MessageSquare className="w-5 h-5" />
                  <span>Inbox</span>
                  {unreadCount > 0 && (
                    <Badge variant="destructive">{unreadCount} unread</Badge>
                  )}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {filteredMessages.length > 0 ? (
                    filteredMessages.map((message) => (
                      <div
                        key={message.id}
                        className={`p-4 border rounded-lg cursor-pointer transition-colors hover:bg-muted/50 ${
                          message.status === 'unread' ? 'border-l-4 border-l-blue-500 bg-blue-50/50' : ''
                        }`}
                        onClick={() => setSelectedMessage(message)}
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex items-start space-x-3 flex-1">
                            <Avatar className="mt-1">
                              <AvatarFallback className={getSenderTypeColor(message.sender.type)}>
                                {message.sender.initials}
                              </AvatarFallback>
                            </Avatar>
                            
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center space-x-2 mb-1">
                                <span className={`font-medium ${message.status === 'unread' ? 'font-bold' : ''}`}>
                                  {message.sender.name}
                                </span>
                                <Badge variant="outline" className="text-xs capitalize">
                                  {message.sender.type}
                                </Badge>
                                {message.priority === 'urgent' && (
                                  <Badge variant="destructive" className="text-xs">Urgent</Badge>
                                )}
                              </div>
                              
                              <h4 className={`text-sm mb-1 truncate ${message.status === 'unread' ? 'font-semibold' : ''}`}>
                                {message.subject}
                              </h4>
                              
                              <p className="text-sm text-muted-foreground truncate">
                                {message.content}
                              </p>
                            </div>
                          </div>
                          
                          <div className="flex flex-col items-end space-y-1">
                            <span className="text-xs text-muted-foreground">
                              {formatTimestamp(message.timestamp)}
                            </span>
                            {message.attachments && (
                              <div className="flex items-center text-xs text-muted-foreground">
                                <PaperclipIcon className="w-3 h-3 mr-1" />
                                {message.attachments}
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="text-center py-12">
                      <MessageSquare className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                      <h3 className="text-lg font-medium text-foreground mb-2">No messages found</h3>
                      <p className="text-muted-foreground">Try adjusting your search or filters.</p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}