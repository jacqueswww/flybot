new Vue({
  el: '#app',
  data: {
    messages: [{
      sender: 'bot', text: 'Hello there!'
    }],
    newMessage: ''
  },
  methods: {
    sendMessage(asBot) {
      if (this.newMessage.trim() !== '') {
        this.messages.push({
          text: this.newMessage,
          sender: asBot ? 'bot' : 'user'
        });
        this.newMessage = '';
      }
    }
  }
});
