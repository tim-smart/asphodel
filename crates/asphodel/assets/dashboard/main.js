// The page's entry: mounts the dashboard into #app with the window's fetch.

import { mount } from "./app.js";

mount(document.getElementById("app"), { fetch: window.fetch.bind(window) });
