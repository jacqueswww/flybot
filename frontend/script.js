new Vue({
  el: '#app',
  data: {
    messages: [{
      sender: 'bot', text: 'Hello there! My name is FlyBot, I am here to assist you with FlySafair bookings and general queries.'
    }],
    newMessage: '',
    socket: null,
    disconnected: false
  },
  mounted() {
    this.connectSocket()
  },
  beforeDestroy() {
    // Close WebSocket connection before the component is destroyed
    if (this.socket) {
      this.socket.close();
    }
  },
  methods: {
    connectSocket() {
      this.disconnected = false;
      let hostURL = String(window.location.origin).replace(window.location.protocol, 'ws:')
      this.socket = new WebSocket(hostURL + `/feed`);
      // Event listener for WebSocket connection established
      this.socket.addEventListener('open', function (event) {
        console.log('WebSocket connection established');
      });
      // Event listener for messages received from the server
      this.socket.addEventListener('message', this.onMessage);
      // Event listener for WebSocket connection closed
      this.socket.addEventListener('close', this.onClose);
      // Event listener for WebSocket connection errors
      this.socket.addEventListener('error', function (event) {
        console.error('WebSocket error:', event);
      });
    },
    markdownToHtml(markdown) {
      // Convert bold text: **text** to <strong>text</strong>
      markdown = markdown.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');    
      // Convert italic text: *text* to <em>text</em>
      markdown = markdown.replace(/\*(.*?)\*/g, '<em>$1</em>');
      // Convert inline code: `code` to <code>code</code>
      markdown = markdown.replace(/`(.*?)`/g, '<code>$1</code>');
      // Convert links: [text](url) to <a href="url">text</a>
      markdown = markdown.replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2">$1</a>');
            // Convert headings: # text to <h1>text</h1>, ## text to <h2>text</h2>, and so on
      markdown = markdown.replace(/#{1,6} (.*?)(?:\n|$)/g, function(match, p1) {
          const level = match.indexOf('#');
          return '<h' + level + '>' + p1.trim() + '</h' + level + '>';
      });
      // Convert paragraphs: text to <p>text</p>
      markdown = markdown.replace(/^.*(?:\n|$)/gm, '<p>$&</p>');
      return markdown;
    },
    onClose(event) {
      console.log('WebSocket connection closed');
      this.disconnected = true
    },
    onMessage(event) {
      console.log('Message from server:', event.data);
      let msg_payload = JSON.parse(event.data)
      let existing_msg = this.messages.find(x => x.id == msg_payload.id)
      console.log(msg_payload)
      if (existing_msg && existing_msg.type == 'action') {
        existing_msg.text += this.markdownToHtml(msg_payload.text);
        existing_msg.type = msg_payload.type;
        if (msg_payload.type == 'output') {
          existing_msg.text = this.markdownToHtml(msg_payload.text)
        }
      } else {
        this.messages.push({
          id: msg_payload.id,
          text: this.markdownToHtml(msg_payload.text),
          type: msg_payload.type,
          sender: 'bot'
        })
      }
    },
    sendMessage(asBot) {
      if (this.newMessage.trim() !== '') {
        let msg_payload = {
          text: this.newMessage,
          sender: asBot ? 'bot' : 'user'
        }
        this.messages.push(msg_payload);
        if (this.socket) {
          this.socket.send(JSON.stringify(msg_payload));
        }
        this.newMessage = '';
      }
    }
  }
});
