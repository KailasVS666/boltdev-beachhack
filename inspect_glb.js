const fs = require('fs');

function inspectGlb(filePath) {
    const buffer = fs.readFileSync(filePath);

    // GLB Header
    const magic = buffer.toString('utf8', 0, 4);
    if (magic !== 'glTF') {
        console.error('Not a valid GLB file');
        return;
    }

    // First chunk is JSON
    const jsonChunkLength = buffer.readUInt32LE(12);
    const jsonChunkType = buffer.toString('utf8', 16, 20);

    if (jsonChunkType !== 'JSON') {
        console.error('First chunk is not JSON');
        return;
    }

    const jsonContent = buffer.toString('utf8', 20, 20 + jsonChunkLength);
    const gltf = JSON.parse(jsonContent);

    console.log('--- GLB Scene Analysis ---');
    console.log(`Total Nodes: ${gltf.nodes ? gltf.nodes.length : 0}`);
    console.log(`Total Meshes: ${gltf.meshes ? gltf.meshes.length : 0}`);
    console.log(`Total Materials: ${gltf.materials ? gltf.materials.length : 0}`);
    console.log('\n--- Node Hierarchy (Groups & Meshes) ---');

    if (gltf.nodes) {
        gltf.nodes.forEach((node, index) => {
            const type = node.mesh !== undefined ? 'MESH' : 'GROUP/EMPTY';
            console.log(`[${index}] ${node.name || 'Unnamed'} (${type})`);
            if (node.children) {
                console.log(`    Children: ${node.children.join(', ')}`);
            }
        });
    }
}

const filePath = process.argv[2];
if (!filePath) {
    console.error('Usage: node inspect_glb.js <path_to_glb>');
} else {
    inspectGlb(filePath);
}
