import "./HealthMonitorMockup.css";
import FallAlertOverlay from "./FallAlertOverlay";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faWifi } from '@fortawesome/free-solid-svg-icons'
import { useEffect, useState } from "react";

const initialData = {
  profile: {
    name: "로딩 중...",
    age: 0,
    status: "확인 중",
    lastActive: "데이터 불러오는 중",
    image: "",
  },

  metrics: {
    walkingSpeed: {
      value: "--",
      unit: "m/s",
      status: "확인 중",
    },

    steps: {
      value: "--",
      unit: "----",
      status: "",
    },

    heartRate: {
      value: "--",
      unit: "bpm",
      status: "확인 중",
    },

    battery: {
      value: "--",
      unit: "%",
      status: "확인 중",
    },
  },

  location: {
    realtime: true,
    mapText: "위치 불러오는 중",
  },

  emergency: {
    showFallAlert: false,
  },
};
const API_BASE = "http://127.0.0.1:8000";

export default function HealthMonitorMockup() {
  const [data, setData] = useState(initialData);
  const [showFallAlert, setShowFallAlert] = useState(false);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === "a" || e.key === "A") {
        setShowFallAlert(true);
      }

      if (e.key === "s" || e.key === "S") {
        setShowFallAlert(false);
      }
    };

    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, []);

  useEffect(() => {
    fetch("http://127.0.0.1:8000/status?user_id=1")
      .then((res) => res.json())
      .then((result) => {
        console.log(result);
        setData(result);

        setShowFallAlert(
          result.emergency?.showFallAlert || false
        );
      })
      .catch((err) => {
        console.error("API Error:", err);
      });
  }, []);

  return (
    <div className="app-bg d-flex align-items-center justify-content-center p-4">
      <div className="phone-frame position-relative overflow-hidden">
        {/* Status bar */}
        <div className="status-bar d-flex align-items-center justify-content-between px-4">
          <span className="fw-bold small">9:41</span>
          <FontAwesomeIcon icon={faWifi} />
        </div>

        <main className="app-body px-4 pb-4">
          {/* Profile */}
          <section className="d-flex align-items-center gap-3 pt-3">
            <img
               src={`${API_BASE}/${data.profile.image}`}
              alt="profile"
              className="profile-img"
            />

            <div>
              <div className="d-flex align-items-end gap-2">
                <h1 className="m-0 fw-black text-dark">{data.profile.name}</h1>
                <span className="age-text fw-bold">{data.profile.age}세</span>
              </div>

              <span className="badge rounded-pill bg-success mt-2 px-3 py-2">
                {data.profile.status}
              </span>

              <p className="text-secondary small fw-semibold mt-2 mb-0">
                {data.profile.lastActive}
              </p>
            </div>
          </section>

          {/* Metric cards */}
        <section className="row g-3 mt-1 align-items-stretch">
          <div className="col-6 d-flex">
            <MetricCard
              icon={<GaugeIcon />}
              value={data.metrics.walkingSpeed.value}
              unit="m/s"
              title="보행 속도"
              status={data.metrics.walkingSpeed.status}
            />
          </div>

          <div className="col-6 d-flex">
            <MetricCard
              icon="👟"
              value={data.metrics.steps.value}
              unit={"/ " + data.metrics.steps.unit}
              title="걸음 수"
              status=""
            />
          </div>

          <div className="col-6 d-flex">
            <MetricCard
              icon="〽"
              value={data.metrics.heartRate.value}
              unit="bpm"
              title="심박수"
              status={data.metrics.heartRate.status}
            />
          </div>

          <div className="col-6 d-flex">
            <MetricCard
              icon="🔋"
              value={data.metrics.battery.value}
              unit="%"
              title="배터리"
              status={data.metrics.battery.status}
            />
          </div>
        </section>

          {/* Map Placeholder */}
          <section className="map-placeholder mt-2">
            <div className="map-header">
              <span className="map-title">현재 위치</span>
              <span className="map-status">실시간</span>
            </div>

            <div className="map-area">
              <span className="map-pin">📍</span>
              <p className="map-text">Google Map 영역</p>
            </div>
          </section>

          <button className="sos-button">
            긴급 SOS
          </button>
        </main>

        {/* Fall Alert Overlay */}
        {showFallAlert && (
            <FallAlertOverlay
                onClose={() => setShowFallAlert(false)}
            />
            )}
      </div>
    </div>
  );
}

function MetricCard({ icon, value, unit, title, status }) {
  return (
    <div className="metric-card p-3 w-100 d-flex flex-column">
      <div className="d-flex justify-content-between align-items-start">
        <div className="metric-icon">{icon}</div>

        <span
          className={`badge bg-light text-secondary rounded-pill ${
            !status ? "invisible" : ""
          }`}
        >
          {status || "빈칸"}
        </span>
      </div>

      <div className="metric-bottom mt-auto">
        <div className="metric-value-row d-flex align-items-baseline gap-1">
          <span className="metric-value fw-black">{value}</span>

          <span className="text-secondary small fw-bold">
            {unit}
          </span>
        </div>

        <p className="text-secondary small fw-semibold mb-0">{title}</p>
      </div>
    </div>
  );
}

function InfoItem({ icon, title, desc }) {
  return (
    <div className="info-item d-flex align-items-center gap-2 p-3">
      <div className="info-icon d-flex align-items-center justify-content-center fw-bold">
        {icon}
      </div>

      <div>
        <p className="small fw-bold mb-0 text-dark">{title}</p>

        <p className="info-desc mb-0 text-secondary">{desc}</p>
      </div>
    </div>
  );
}

function GaugeIcon() {
  return (
    <svg width="42" height="42" viewBox="0 0 42 42" fill="none">
      <path
        d="M8 27a13 13 0 1 1 26 0"
        stroke="currentColor"
        strokeWidth="5"
        strokeLinecap="round"
      />

      <path
        d="M21 27l9-9"
        stroke="#22c55e"
        strokeWidth="3"
        strokeLinecap="round"
      />
    </svg>
  );
}