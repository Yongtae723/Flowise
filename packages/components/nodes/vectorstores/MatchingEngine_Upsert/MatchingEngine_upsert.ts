import { ICommonObject, INode, INodeData, INodeOutputsValue, INodeParams } from '../../../src/Interface'
import { MatchingEngine, MatchingEngineArgs } from "langchain/vectorstores/googlevertexai";
import { Document } from "langchain/document";
import { GoogleCloudStorageDocstore} from "langchain/stores/doc/gcs";
import { Embeddings } from 'langchain/embeddings/base'
import { getBaseClasses, getCredentialData, getCredentialParam } from '../../../src/utils'
import { GoogleAuthOptions } from 'google-auth-library'
import { flatten } from 'lodash'

class MatchingEngineUpsert_VectorStores implements INode {
    label: string
    name: string
    version: number
    description: string
    type: string
    icon: string
    category: string
    baseClasses: string[]
    inputs: INodeParams[]
    credential: INodeParams
    outputs: INodeOutputsValue[]

    constructor() {
        this.label = 'Matching Engine Upsert Document'
        this.name = 'vertexaiUpsert'
        this.version = 1.0
        this.type = 'MatchingEngine'
        this.icon = 'vertexai.svg'
        this.category = 'Vector Stores'
        this.description = 'Upsert documents to Matching Engine'
        this.baseClasses = [this.type, 'VectorStoreRetriever', 'BaseRetriever']
        this.credential = {
            label: 'Connect Credential',
            name: 'credential',
            type: 'credential',
            credentialNames: ['googleVertexAuth']
        }
        this.inputs = [
            {
                label: 'Document',
                name: 'document',
                type: 'Document',
                list: true
            },
            {
                label: 'Embeddings',
                name: 'embeddings',
                type: 'Embeddings'
            },
            {
                label: 'DocsBucket',
                name: 'bucket',
                type: 'string',
                placeholder: 'gs://my-bucket',
            },
            {
                label: 'Index',
                name: 'index',
                type: 'string'
            },
            {
                label: 'Index Endpoint',
                name: 'indexEndpoint',
                type: 'string'
            },
            {
                label: 'API Version',
                name: 'apiVersion',
                type: 'string',
                placeholder: 'v1beta1',
                additionalParams: true,
            },
            {
                label: 'Top K',
                name: 'topK',
                description: 'Number of top results to fetch. Default to 4',
                placeholder: '4',
                type: 'number',
                additionalParams: true,
                optional: true
            }
        ]
        this.outputs = [
            {
                label: 'Matching Engine Retriever',
                name: 'retriever',
                baseClasses: this.baseClasses
            },
            {
                label: 'Matching Engine Vector Store',
                name: 'vectorStore',
                baseClasses: [this.type, ...getBaseClasses(MatchingEngine)]
            }
        ]
    }

    async init(nodeData: INodeData, _: string, options: ICommonObject): Promise<any> {
        const index = nodeData.inputs?.pineconeIndex as string
        const docs = nodeData.inputs?.document as Document[]
        const embeddings = nodeData.inputs?.embeddings as Embeddings
        const output = nodeData.outputs?.output as string
        const bucket = nodeData.outputs?.bucket as string
        const topK = nodeData.inputs?.topK as string
        const indexEndpoint = nodeData.inputs?.endpoint as string
        const apiVersion = nodeData.inputs?.apiVersion as string
        const k = topK ? parseFloat(topK) : 4

        const credentialData = await getCredentialData(nodeData.credential ?? '', options)
        const skipExtraCredentialFile = getCredentialParam('skipExtraCredentialFile', credentialData, nodeData)
        const googleApplicationCredentialFilePath = getCredentialParam('googleApplicationCredentialFilePath', credentialData, nodeData)
        const googleApplicationCredential = getCredentialParam('googleApplicationCredential', credentialData, nodeData)
        const projectID = getCredentialParam('projectID', credentialData, nodeData)

        if (!skipExtraCredentialFile && !googleApplicationCredentialFilePath && !googleApplicationCredential)
            throw new Error('Please specify your Google Application Credential')

        const inputs = [googleApplicationCredentialFilePath, googleApplicationCredential, skipExtraCredentialFile]

        if (inputs.filter(Boolean).length > 1) {
            throw new Error(
                'Error: More than one component has been inputted. Please use only one of the following: Google Application Credential File Path, Google Credential JSON Object, or Skip Extra Credential File.'
            )
        }

        const authOptions: GoogleAuthOptions = {}
        if (!skipExtraCredentialFile) {
            if (googleApplicationCredentialFilePath && !googleApplicationCredential)
                authOptions.keyFile = googleApplicationCredentialFilePath
            else if (!googleApplicationCredentialFilePath && googleApplicationCredential)
                authOptions.credentials = JSON.parse(googleApplicationCredential)
            if (projectID) authOptions.projectId = projectID
        }

        const flattenDocs = docs && docs.length ? flatten(docs) : []
        const finalDocs = []
        for (let i = 0; i < flattenDocs.length; i += 1) {
            finalDocs.push(new Document(flattenDocs[i]))
        }

        const store = new GoogleCloudStorageDocstore({
            bucket: bucket,
        });
        const config:MatchingEngineArgs = {
            index: index,
            indexEndpoint: indexEndpoint,
            apiVersion: apiVersion,
            docstore: store,
        };
        if (authOptions) config.authOptions = authOptions

        const vectorStore = await MatchingEngine.fromDocuments(finalDocs, embeddings, config)

        if (output === 'retriever') {
            const retriever = vectorStore.asRetriever(k)
            return retriever
        } else if (output === 'vectorStore') {
            ;(vectorStore as any).k = k
            return vectorStore
        }
        return vectorStore
    }
}

module.exports = { nodeClass: MatchingEngineUpsert_VectorStores }
