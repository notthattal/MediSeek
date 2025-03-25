import { Fragment } from "react"
import MessageItem from "../MessageItem/MessageItem"
import styles from "./MessageList.module.css"
import 'bootstrap-icons/font/bootstrap-icons.css';

const MessageList = ({ messages, isLoading }) => {
  return (
  <div className={styles.messageList}>
    {messages.length === 0 ? (
      <div className={styles.emptyChat}>
        <p><b>Developers' Note:</b> MediSeek should only be used for informational purposes and
        is not meant to replace medical advice. This chatbot is not used for business purposes. It is only meant to further
        the Huberman Lab mission of “zero cost” education for the
        masses. We are consistently training the model on more health podcasts and academic resources for better results.</p>
        <p>Please click the info button or <a href="./evaluation_results/evaluation_report.html" target="_noblank"><nobr>here</nobr></a> to learn more about the models' performances. </p>
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
        <span>MediSeek is looking for an answer...</span>
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

