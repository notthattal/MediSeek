"use client"

import { useState, useRef, useEffect } from "react"
import { Plus, Search, Lightbulb, Mic, Send } from "lucide-react"
import styles from "./ChatInput.module.css"

const ChatInput = ({ onSendMessage, isLoading }) => {
  const [input, setInput] = useState("")
  const textareaRef = useRef(null)

  // Auto-resize the textarea based on content
  useEffect(() => {
    if (textareaRef.current) {
      // Reset height to auto to get the correct scrollHeight
      textareaRef.current.style.height = "auto"
      // Set the height to scrollHeight to expand the textarea
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`
    }
  }, [input])

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!input.trim()) return

    onSendMessage(input)
    setInput("")
    
    // Reset textarea height after submission
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto"
    }
  }

  // Handle Enter key to submit, but allow Shift+Enter for new lines
  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  return (
    <form onSubmit={handleSubmit} className={styles.form}>
      <div className={styles.inputContainer}>
        <button type="button" className={styles.iconButton}>
          <Plus size={20} />
        </button>

        <textarea
          ref={textareaRef}
          className={styles.input}
          placeholder="Ask anything"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isLoading}
          rows={1}
        />

        <div className={styles.actions}>
          {/* If we want to implement text to speech for prompting */}
          <button type="button" className={styles.iconButton}>
            <Mic size={20} />
          </button>
          
          {input.trim() && (
            <button type="submit" className={styles.iconButton} disabled={isLoading}>
              <Send size={20} />
            </button>
          )}
        </div>
      </div>
    </form>
  )
}

export default ChatInput