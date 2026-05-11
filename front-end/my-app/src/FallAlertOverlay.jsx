import "./FallAlertOverlay.css";

export default function FallAlertOverlay({ onClose }) {
  return (
    <div className="fall-overlay d-flex flex-column justify-content-center align-items-center text-center text-white">
      {/* Close Button */}
      <button
        className="close-alert-btn"
        onClick={onClose}
      >
        ✕
      </button>

      <div className="fall-icon mb-4">⚠</div>

      <h1 className="fw-black mb-3">낙상 감지</h1>

      <p className="fall-desc mb-1">
        김영희 님이 넘어졌을 가능성이 있습니다.
      </p>

      <p className="fall-time mb-4">
        2026.05.11 · 오후 8:42
      </p>

      <div className="fall-location mb-4 px-4 py-3 rounded-pill">
        📍 전남대학교 생활관 B동
      </div>

      <button className="sos-button">
        119 긴급 신고
      </button>
    </div>
  );
}