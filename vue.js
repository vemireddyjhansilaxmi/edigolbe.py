const { createApp } = Vue;

createApp({
    data() {
        return {
            demographics: {},
            recoveryStats: {},
            modelStatus: {},
            prediction: null,
            loading: false,
            age: 45,
            contact: 2,
            order: 3
        }
    },
    
    mounted() {
        this.loadData();
    },
    
    methods: {
        async loadData() {
            try {
                const [demoRes, outcomesRes, healthRes] = await Promise.all([
                    fetch('/api/demographics'),
                    fetch('/api/outcomes'),
                    fetch('/api/health')
                ]);
                
                this.demographics = await demoRes.json();
                this.recoveryStats = await outcomesRes.json();
                this.modelStatus = await healthRes.json();
                
            } catch (error) {
                console.error('Load error:', error);
            }
        },
        
        async predictRecovery() {
            this.loading = true;
            this.prediction = null;
            
            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        age: this.age,
                        contact_number: this.contact,
                        infection_order: this.order
                    })
                });
                
                const result = await response.json();
                this.prediction = result;
                
            } catch (error) {
                this.prediction = { error: error.message };
            } finally {
                this.loading = false;
            }
        }
    }
}).mount('#app');
