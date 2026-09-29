self.addEventListener("install",e=>self.skipWaiting());
self.addEventListener("fetch",e=>{if(e.request.method!=="GET")return;e.respondWith(fetch(e.request).then(r=>{const c=r.clone();caches.open("onioniq-v2").then(k=>k.put(e.request,c));return r}).catch(()=>caches.match(e.request)))});
