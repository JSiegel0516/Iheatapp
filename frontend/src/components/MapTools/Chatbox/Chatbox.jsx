import React, { useState, useEffect, useContext } from "react";
import { AppContext } from "../../../Context";

import {
  Widget,
  addResponseMessage,
  toggleMsgLoader,
} from "react-chat-widget-react-18";
import "react-chat-widget-react-18/lib/styles.css";
import ScreenshotButton from "./Screenshot";
import "./Chatbox.css";

import { useTranslation } from "react-i18next";
import i18n from "../../../i18n";

// change this later to get acutal WBGT point value from app context


const AIChat = ({ applicationRef, onOpenChange }) => {
  const { selectedWBGTValue } = useContext(AppContext);
  const wbgtValue =
    selectedWBGTValue !== null
      ? ((selectedWBGTValue - 273.15) * 9) / 5 + 32
      : null;
  const [welcomeMessageShown, setWelcomeMessageShown] = useState(false);
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);

  const { t } = useTranslation();


  const determineCategory = (valueF) => {
    if (valueF === null || valueF === undefined) return null;
    if (valueF < 72) return "Safe";
    if (valueF < 76) return "Caution";
    if (valueF < 80) return "Warning";
    if (valueF < 84) return "Danger";
    return "Extreme";
  };

  const category = determineCategory(wbgtValue);

  useEffect(() => {
    console.log("AIChat component mounted");
  }, []);

  useEffect(() => {
    const handleLanguageChange = () => {
      if (welcomeMessageShown) {
        addResponseMessage(t("Welcome Message"));
      }
    };

    i18n.on("languageChanged", handleLanguageChange);

    return () => {
      i18n.off("languageChanged", handleLanguageChange);
    };
  }, [t, welcomeMessageShown]);

  const handleWidgetToggle = (isOpen) => {
    if (typeof onOpenChange === "function") {
      onOpenChange(isOpen);
    }
    setIsChatOpen(isOpen);

    // Toggle a body class so mobile CSS can react when chat is open
    const body = document?.body;
    if (body) {
      if (isOpen) {
        body.classList.add("chat-open");
      } else {
        body.classList.remove("chat-open");
      }
    }
    if (isOpen && !welcomeMessageShown) {
      addResponseMessage(t("Welcome Message"));
      setWelcomeMessageShown(true);
    }
  };

  // Ensure body class is removed on unmount
  useEffect(() => {
    return () => {
      document?.body?.classList?.remove("chat-open");
    };
  }, []);

  const handleNewUserMessage = async (newMessage, imageDataUrl = null) => {
    if (isProcessing) return;

    try {
      setIsProcessing(true);
      toggleMsgLoader();

      const response = await fetch(`${process.env.REACT_APP_API_URL || "/api"}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: newMessage,
          ...(typeof imageDataUrl === "string" ? { image_data_url: imageDataUrl } : {}),
          language: i18n.language || "en",
          history: chatHistory.slice(-20),
          map_context: {
            wbgt_f: wbgtValue,
            category,
          },
        }),
      });

      const result = await response.json().catch(() => null);
      if (!response.ok) {
        const detail = typeof result?.detail === "string" ? result.detail : `HTTP ${response.status}`;
        throw new Error(detail);
      }

      const reply = result?.answer;
      if (typeof reply !== "string" || !reply.trim()) {
        throw new Error("The assistant returned an empty response.");
      }

      const citations = Array.isArray(result?.citations)
        ? result.citations.filter((citation) => citation?.url)
        : [];
      const sourceList = citations
        .map(
          (citation) =>
            `[${citation.number}] ${citation.title}: ${citation.url}`
        )
        .join("\n");
      const displayedReply = sourceList
        ? `${reply}\n\nSources:\n${sourceList}`
        : reply;

      addResponseMessage(displayedReply);
      setChatHistory((history) => [
        ...history,
        { role: "user", content: newMessage },
        { role: "assistant", content: reply },
      ]);
    } catch (error) {
      console.error("Chat error:", error);
      addResponseMessage(
        `Sorry, Sol could not answer right now. ${error.message}`
      );
    } finally {
      setIsProcessing(false);
      toggleMsgLoader();
    }
  };

  const handleScreenshotCaptured = async (imageDataUrl) => {
    await handleNewUserMessage(
      "Interpret the displayed heatmap and selected WBGT point, and explain relevant heat-safety precautions.",
      imageDataUrl
    );
  };
  return (
    <>
      <Widget
        handleNewUserMessage={handleNewUserMessage}
        handleToggle={handleWidgetToggle}
        title="Sol"
        subtitle={t("Your WBGT data AI assistant")}
      />

      <div className="screenshot-button-container">
        <ScreenshotButton
          applicationRef={applicationRef}
          onCapture={handleScreenshotCaptured}
          isProcessing={isProcessing}
        />
      </div>
    </>
  );
};

export default AIChat;
