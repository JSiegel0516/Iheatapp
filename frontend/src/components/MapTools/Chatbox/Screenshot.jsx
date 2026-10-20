import React, { useState } from "react";
import html2canvas from "html2canvas";
import { BiCamera } from "react-icons/bi";
import { toggleWidget } from "react-chat-widget-react-18";

const ScreenshotButton = ({ applicationRef, onCapture, isProcessing }) => {
  const [isCapturing, setIsCapturing] = useState(false);

  const captureScreenshot = async () => {
    if (isCapturing || isProcessing) return;

    if (!applicationRef || !applicationRef.current) {
      alert(
        "Unable to capture screenshot: application reference not provided."
      );
      return;
    }

    let closedChat = false;
    try {
      setIsCapturing(true);

      // Close the chatbox if open
      const chatWidget = document.querySelector(".rcw-widget-container");
      if (chatWidget && chatWidget.classList.contains("rcw-opened")) {
        toggleWidget();
        closedChat = true;
      }

      // Always wait a bit to allow layout changes (e.g., chat closing) to settle
      await new Promise((resolve) =>
        setTimeout(resolve, closedChat ? 300 : 100)
      );

      const canvas = await html2canvas(applicationRef.current, {
        useCORS: true,
        allowTaint: false,
        logging: false,
        scale: Math.min(1, 1600 / Math.max(applicationRef.current.clientWidth, applicationRef.current.clientHeight, 1)),
        ignoreElements: (element) =>
          element.classList.contains("rcw-widget-container") ||
          element.classList.contains("screenshot-button-container"),
      });

      const imageDataUrl = canvas.toDataURL("image/png");
      if (closedChat) {
        toggleWidget();
        closedChat = false;
      }
      await onCapture(imageDataUrl);
    } catch (error) {
      console.error("Error capturing screenshot:", error);
      alert("Failed to capture screenshot. Please try again.");
    } finally {
      if (closedChat) toggleWidget();
      setIsCapturing(false);
    }
  };

  // Determine if button should be disabled
  const isDisabled = isCapturing || isProcessing;

  return (
    <button
      type="button"
      aria-label="Analyze a screenshot of the heatmap"
      onClick={captureScreenshot}
      className={`screenshot-button ${
        isDisabled ? "screenshot-button-disabled" : ""
      }`}
      disabled={isDisabled}
    >
      {isCapturing ? (
        "Capturing..."
      ) : (
        <BiCamera size={24} color={isDisabled ? "#cccccc" : "white"} />
      )}
    </button>
  );
};

export default ScreenshotButton;
