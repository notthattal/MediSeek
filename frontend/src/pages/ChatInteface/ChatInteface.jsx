"use client"

import { useState } from "react"
import MessageList from "../../components/MessageList/MessageList"
import ChatInput from "../../components/ChatInput/ChatInput"
import styles from "./ChatInterface.module.css"

const ChatInterface = () => {
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)

  const handleSendMessage = (messageText) => {
    if (!messageText.trim()) return

    const userMessage = {
      id: Date.now(),
      content: messageText,
      role: "user",
    }

    setMessages((prevMessages) => [...prevMessages, userMessage])
    setIsLoading(true)

    // Simulate response (replace with your API)
    setTimeout(() => {
      const aiMessage = {
        id: Date.now() + 1,
        content: `This is a simulated response to: "${messageText}"`,
        role: "assistant",
      }

      setMessages((prevMessages) => [...prevMessages, aiMessage])
      setIsLoading(false)
    }, 1500)
  }

  return (
    <div className={styles.container}>
      <div className={styles.chatContainer}>
        <h1 className={styles.heading}>What can I help with?</h1>

        <div className={styles.messageArea}>
          <MessageList messages={messages} isLoading={isLoading} />
        </div>

        <div className={styles.inputArea}>
          <ChatInput onSendMessage={handleSendMessage} isLoading={isLoading} />
          <div className={styles.quickActions}>
            <button className={styles.actionButton}>
              <span className={styles.icon}>🎨</span>
              Create image
            </button>
            <button className={styles.actionButton}>
              <span className={styles.icon}>✍️</span>
              Help me write
            </button>
            <button className={styles.actionButton}>
              <span className={styles.icon}>🎲</span>
              Surprise me
            </button>
            <button className={styles.actionButton}>
              <span className={styles.icon}>📝</span>
              Summarize text
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ChatInterface

