// Positions are in the table SVG's coordinate system, not device pixels.
window.initCuePositionEditor = (hostId, initial) => {
    const host = document.getElementById(hostId);
    const viewport = host.querySelector('.position-viewport');
    const svg = viewport.querySelector('svg');
    let center = [...initial], scale = 1;
    let origin;
    const pointers = new Map();
    const paint = () => {
        const w = viewport.clientWidth / scale, h = viewport.clientHeight / scale;
        svg.setAttribute('viewBox', `${center[0] - w / 2} ${center[1] - h / 2} ${w} ${h}`);
    };
    const gesture = () => {
        const p = [...pointers.values()];
        return {
            x: p.reduce((s, v) => s + v.x, 0) / p.length,
            y: p.reduce((s, v) => s + v.y, 0) / p.length,
            distance: p.length === 2 ? Math.hypot(p[0].x - p[1].x, p[0].y - p[1].y) : 0,
        };
    };
    const rebase = () => {
        origin = pointers.size ? {...gesture(), center: [...center], scale} : null;
    };
    viewport.addEventListener('pointerdown', e => {
        if (pointers.size >= 2) return;
        viewport.setPointerCapture(e.pointerId);
        pointers.set(e.pointerId, {x: e.clientX, y: e.clientY});
        rebase();
    });
    viewport.addEventListener('pointermove', e => {
        if (!pointers.has(e.pointerId)) return;
        pointers.set(e.pointerId, {x: e.clientX, y: e.clientY});
        const after = gesture();
        center[0] = origin.center[0] - (after.x - origin.x) / origin.scale;
        center[1] = origin.center[1] - (after.y - origin.y) / origin.scale;
        if (origin.distance > 0)
            scale = Math.max(0.4, Math.min(6, origin.scale * after.distance / origin.distance));
        paint();
    });
    for (const event of ['pointerup', 'pointercancel', 'lostpointercapture'])
        viewport.addEventListener(event, e => { pointers.delete(e.pointerId); rebase(); });
    viewport.addEventListener('wheel', e => {
        e.preventDefault();
        scale = Math.max(0.4, Math.min(6, scale * Math.exp(-e.deltaY * 0.002)));
        paint();
    }, {passive: false});
    host.positionEditor = {
        point: () => [...center],
        zoom: factor => { scale = Math.max(0.4, Math.min(6, scale * factor)); paint(); },
    };
    const observer = new ResizeObserver(paint);
    observer.observe(viewport);
    host.positionEditor.dispose = () => observer.disconnect();
    paint();
};
