// Generate a unique session ID for this chat session
let sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
let isLoading = false;

const chatContainer = document.getElementById('chatContainer');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const newChatBtn = document.getElementById('newChatBtn');

// Auto-resize textarea
messageInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});

// Send message on Enter (Shift+Enter for new line)
messageInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Send button click
sendBtn.addEventListener('click', sendMessage);

// New chat button
newChatBtn.addEventListener('click', startNewChat);

async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message || isLoading) return;

    // Add user message to chat
    addMessage('user', message);
    messageInput.value = '';
    messageInput.style.height = 'auto';
    
    // Show loading indicator
    const loadingId = addLoadingMessage();
    isLoading = true;
    sendBtn.disabled = true;

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: message,
                session_id: sessionId
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        // Debug: Check the raw response
        console.log('=== DEBUG: Response received ===');
        const answer = data.answer || 'Sorry, I could not generate a response.';
        console.log('Answer type:', typeof answer);
        console.log('Answer length:', answer.length);
        console.log('Has newline (\\n):', answer.includes('\n'));
        console.log('Has carriage return (\\r):', answer.includes('\r'));
        console.log('First 500 characters:', answer.substring(0, 500));
        
        // Check for newline character codes
        const newlineCheck = [];
        for (let i = 0; i < Math.min(answer.length, 500); i++) {
            const charCode = answer.charCodeAt(i);
            if (charCode === 10 || charCode === 13) {
                newlineCheck.push({ index: i, code: charCode, char: charCode === 10 ? '\\n' : '\\r' });
            }
        }
        console.log('Newline characters found:', newlineCheck);
        console.log('=== END DEBUG ===');
        
        // Remove loading indicator
        removeLoadingMessage(loadingId);
        
        // Add assistant response (preserve newlines from response)
        addMessage('assistant', answer);
    } catch (error) {
        console.error('Error:', error);
        removeLoadingMessage(loadingId);
        addMessage('assistant', 'Sorry, an error occurred. Please try again.', true);
    } finally {
        isLoading = false;
        sendBtn.disabled = false;
        messageInput.focus();
    }
}

function addMessage(role, content, isError = false) {
    // Remove welcome message if it exists
    const welcomeMsg = chatContainer.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    if (isError) {
        messageDiv.classList.add('error-message');
    }

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // Convert newlines to <br> tags while escaping HTML to prevent XSS
    // First, ensure content is a string
    let textContent = String(content);
    
    // Debug: Check for newlines (temporary)
    if (role === 'assistant') {
        console.log('Content length:', textContent.length);
        console.log('Has \\n:', textContent.includes('\n'));
        console.log('Has \\r\\n:', textContent.includes('\r\n'));
        console.log('First 300 chars:', textContent.substring(0, 300));
        // Check char codes around where newlines should be
        const sample = textContent.substring(0, 500);
        const newlineIndices = [];
        for (let i = 0; i < sample.length; i++) {
            const code = sample.charCodeAt(i);
            if (code === 10 || code === 13) {
                newlineIndices.push(i);
            }
        }
        console.log('Newline char codes found at indices:', newlineIndices);
    }
    
    // Escape HTML special characters to prevent XSS
    // With white-space: pre-line CSS, we can preserve newlines directly
    let escapedContent = textContent
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
    
    // Use textContent instead of innerHTML since CSS white-space: pre-line will handle newlines
    // This is safer and simpler
    contentDiv.textContent = textContent;

    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    timeDiv.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    messageDiv.appendChild(contentDiv);
    messageDiv.appendChild(timeDiv);
    chatContainer.appendChild(messageDiv);

    // Scroll to bottom
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function addLoadingMessage() {
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message assistant';
    loadingDiv.id = 'loading-message';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content loading';
    contentDiv.innerHTML = `
        <span>Thinking</span>
        <span class="loading-dot"></span>
        <span class="loading-dot"></span>
        <span class="loading-dot"></span>
    `;

    loadingDiv.appendChild(contentDiv);
    chatContainer.appendChild(loadingDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    return 'loading-message';
}

function removeLoadingMessage(id) {
    const loadingMsg = document.getElementById(id);
    if (loadingMsg) {
        loadingMsg.remove();
    }
}

async function startNewChat() {
    if (isLoading) return;

    // Clear chat history on server
    try {
        await fetch('/new-chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: sessionId
            })
        });
    } catch (error) {
        console.error('Error clearing chat:', error);
    }

    // Generate new session ID
    sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    // Clear chat UI
    chatContainer.innerHTML = `
        <div class="welcome-message">
            <p>Welcome! I can help you analyze the California procurement dataset. Ask me anything!</p>
            <p class="attribution">Project by Abdullah Aktam Anjrini for Penny Software as part of an assessment.</p>
        </div>
    `;
    
    messageInput.focus();
}

// Focus input on load
messageInput.focus();

