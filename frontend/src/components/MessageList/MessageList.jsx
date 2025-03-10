import { Fragment } from "react"
import MessageItem from "../MessageItem/MessageItem"
import styles from "./MessageList.module.css"
import 'bootstrap-icons/font/bootstrap-icons.css';

const MessageList = ({ messages, isLoading }) => {
  return (
    <div className={styles.messageList}>
      {messages.length === 0 ? (
        <div className={styles.emptyChat}>
          <p>👋 Hello! How can I help you today?</p>
        </div>
      ) : (
        <Fragment>
          {messages.map((message) => (
            <MessageItem key={message.id} message={message} />
          ))}
        </Fragment>
      )}

      {isLoading && (
        <div className={`d-flex justify-content-start mb-4 ${styles.loadingContainer}`}>
          <div className={styles.avatar}>
            <span>AI</span>
          </div>
          <div className={`${styles.messageBubble} ${styles.aiMessage}`}>
            <div className={styles.typingIndicator}>
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default MessageList

