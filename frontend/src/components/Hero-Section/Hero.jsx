import "./Hero.css"; 
import { useState, useEffect, useRef } from "react";
import { File, Send, Bot, User, X } from "lucide-react";
import ReactMarkdown from "react-markdown";

const Hero = () => {
  const API_URL = "http://127.0.0.1:8000"
  const [text, setText] = useState("");
  const textarea = useRef(null);
  const chatHistoryRef = useRef(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [messages, setMessages] = useState([
    {
      id: 1,
      content: "Hello! I'm your Legal Navigator AI. How can I assist you today?",
      sender: "ai",
    },
  ]);

  useEffect(() => {
    if (textarea.current) {
      textarea.current.style.height = "auto";
      const scrollHeight = textarea.current.scrollHeight;
      textarea.current.style.height = scrollHeight + "px";
    }
  }, [text]);

  useEffect(() => {
    if (chatHistoryRef.current) {
      const { scrollHeight } = chatHistoryRef.current;
      chatHistoryRef.current.scrollTop = scrollHeight;
    }
  }, [messages]);

  const handleChange = (event) => {
    setText(event.target.value);
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  const handleRemoveFile = async () => {
    setSelectedFile(null);
    setPreviewUrl("");
    const fileInput = document.getElementById("file-upload");
    if (fileInput) fileInput.value = "";
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage(e);
    }
  };

  const handleSendMessage = async (event) => {
    event.preventDefault();
    if (text.trim() === "" && !selectedFile) return;

    const userMessage = {
      id: Date.now(),
      content: text,
      sender: localStorage.getItem("username") || "user",
      previewUrl: previewUrl,
      selectedFileName: selectedFile?.name,
      selectedFileType: selectedFile?.type,
    };

    setMessages((prevMessages) => [...prevMessages, userMessage]);

    setText("");
    handleRemoveFile();
    try {
      const formData = new FormData();
      const sessionId = sessionStorage.getItem("session_id");
      if (sessionId) {
        formData.append("session_id", sessionId);
      }

      formData.append("message", text);
      formData.append("sender", userMessage.sender);
      if (selectedFile) {
        formData.append("file", selectedFile);
      }
      const res = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      console.log(data);

      if (data.session_id) {
        sessionStorage.setItem("session_id", data.session_id);
      }
      const aiMessage = {
        id: Date.now() + 1,
        content: data.reply,
        sender: "ai",
      };
      setMessages((prev) => [...prev, aiMessage]);
    } catch (error) {
      console.log("Error sending message:", error);
    }
  };

  return (
    <>
      <div className="hero-section">
        <div className="chat-history" ref={chatHistoryRef}>
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`message ${msg.sender === "ai" ? "ai" : "user"}`}
            >
              {msg.previewUrl && (
                <>
                  {msg.selectedFileType?.startsWith("image/") ? (
                    <img
                      src={msg.previewUrl}
                      alt="attachment"
                      className="message-image"
                    />
                  ) : msg.selectedFileType === "application/pdf" ? (
                    <embed
                      src={msg.previewUrl}
                      type="application/pdf"
                      width="250"
                      height="200"
                      className="message-file"
                    />
                  ) : (
                    <a
                      href={msg.previewUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="file-link"
                    >
                      📎 {msg.selectedFileName}
                    </a>
                  )}
                </>
              )}

              
              {msg.sender === "ai" ? <Bot color="#4A90E2" /> : ""}
              {msg.content && <ReactMarkdown>{msg.content}</ReactMarkdown>}
              {msg.sender === "ai" ? "" : <User size={20} color="#14161A" />}
            </div>
          ))}
        </div>

        {selectedFile && (
          <div className="preview-container">
            {selectedFile.type.startsWith("image/") ? (
              <img src={previewUrl} alt="preview" className="preview-image" />
            ) : selectedFile.type === "application/pdf" ? (
              <embed
                src={previewUrl}
                type="application/pdf"
                width="200"
                height="150"
                className="preview-file"
              />
            ) : (
              <span className="preview-file-name">📎 {selectedFile.name}</span>
            )}
            <button className="remove-btn" onClick={handleRemoveFile}>
              <X size={18} />
            </button>
          </div>
        )}

        <form onSubmit={handleSendMessage}>
          <label htmlFor="file-upload" className="file-upload-label">
            <div className="plus-mark">
              <File size={20} color="#4A90E2" />
            </div>
          </label>
          <input
            type="file"
            id="file-upload"
            style={{ display: "none" }}
            onChange={handleFileChange}
          />
          <div className="input-wrapper">
            <textarea
              ref={textarea}
              value={text}
              onChange={handleChange}
              onKeyDown={handleKeyDown}
              placeholder="Enter your message..."
              rows={1}
            ></textarea>
          </div>
          <button type="submit" className="send-btn">
            <Send size={20} color="black" />
          </button>
        </form>
      </div>
    </>
  );
};

export default Hero;
