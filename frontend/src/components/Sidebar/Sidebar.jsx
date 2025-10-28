import { useState } from 'react';
import './Sidebar.css'
import { MessageSquare, Plus } from 'lucide-react';
const Sidebar = () => {

  const [chat, setchat] = useState([
    "How to file a PIL?", "Bail application", "Draft a notice"
  ])
  return (
    <>
     <div className="sidebar">
        <div className="top-section">
            <MessageSquare className='msgsq' size={24} color="#4BE784"/>
            <span>Legal Chat Assistant</span>
        </div>
        <div className="new-chat">
          <button type="button">
            <Plus size={14} color="#000"/>
            <span>New chat</span>
          </button>
        </div>
        <div className="history">
          <span>Recent Chats</span>
          <ul>
            {chat.map((item, index) => (
              <li key={index}>
                  <button type='button'>
                <MessageSquare size={14} color="#fff"/>
                <span>{item}</span>
              </button>
              </li>
            ))}
          </ul>
        </div>
     </div>
    </>
  )
}

export default Sidebar