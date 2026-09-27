import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  vus: 15,
  duration: "60s",
  thresholds: {
    http_req_failed: ["rate<0.01"],
    http_req_duration: ["p(95)<2000"],
  },
};

const BASE = __ENV.API_URL || "http://localhost:8000";

export default function () {
  const res = http.post(
    `${BASE}/api/predict`,
    JSON.stringify({
      features: {
        age: 30,
        sex: "male",
        bmi: 25.5,
        children: 2,
        smoker: "no",
        region: "southeast",
      },
    }),
    {
      headers: {
        "Content-Type": "application/json",
        "X-Request-ID": `k6-${__VU}-${__ITER}`,
      },
    }
  );
  check(res, { "status 200": (r) => r.status === 200 });
  sleep(0.3);
}
