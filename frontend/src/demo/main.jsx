import React from "react";
import ReactDOM from "react-dom/client";

import "../shared.css";
import DemoApp from "./DemoApp";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <DemoApp />
  </React.StrictMode>,
);
